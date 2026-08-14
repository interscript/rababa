"""Dataset loaders — Tashkeela++ Arabic, Nakdimon Hebrew.

Each loader returns a `Dataset` object with parallel pairs
(input: undiacritized, output: haraqat-IDs).

Tashkeela format (local files): one line per example, fully diacritized
Arabic text. We extract `(letters, haraqat)` per character position.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from .constants import (
    ALL_POSSIBLE_HARAQAT,
    INPUT_VOCAB_SIZE,
    MASK_ID,
    PAD_ID,
    TARGET_VOCAB,
)
from .constants_hebrew import (
    DAGESH_VOCAB,
    HEBREW_LETTERS,
    INPUT_VOCAB_SIZE as HEBREW_INPUT_VOCAB_SIZE,
    MASK_ID as HEBREW_MASK_ID,
    NIQQUD_VOCAB,
    SIN_VOCAB,
    can_dagesh,
    can_niqqud,
    can_sin,
    is_hebrew_letter,
)
from .encoder import ArabicEncoder, HebrewEncoder

# Default corpus location — Tashkeela (shipped with repo).
DEFAULT_TASHKEELA_ROOT = Path(__file__).resolve().parent.parent.parent / "test-datasets" / "tashkeela"

# MLM pretrain corpus — prefer the larger Wikipedia dump if available.
def _find_arabic_mlm_root() -> Path:
    candidates = [
        Path("/opt/rababa/data/arwiki"),    # Modal image build-time clone
        Path("/datasets/arwiki"),            # Legacy volume mount
        Path(__file__).resolve().parent.parent.parent / "data" / "arwiki",  # Local dev
        DEFAULT_TASHKEELA_ROOT,              # Fall back to Tashkeela undiacritized.
    ]
    for c in candidates:
        if (c / "train.txt").is_file():
            return c
    return candidates[-1]


ARABIC_MLM_ROOT = _find_arabic_mlm_root()

# Lookups for splitting diacritized text into (letters, haraqat).
_HARAQAT_TO_ID = {h: i for i, h in enumerate(TARGET_VOCAB[1:-1])}  # pad + extra slot excluded
_HARAQAT_CHARS = set(ALL_POSSIBLE_HARAQAT.keys())


@dataclass(frozen=True)
class Example:
    """One parallel (input, target) pair after encoding."""

    input_ids: list[int]
    target_ids: list[int]
    raw: str  # for debugging / eval


def _extract_pairs(diacritized: str) -> tuple[list[str], list[str]]:
    """Split diacritized text into (letters, haraqat-strings).

    Each Arabic letter is followed by 0+ haraqat. We group them:
    the letter goes into `letters`, all subsequent haraqat concat into
    the matching `haraqat` slot.
    """
    letters: list[str] = []
    haraqat: list[str] = []
    current_h = ""
    for ch in diacritized:
        if ch in _HARAQAT_CHARS and ch:
            current_h += ch
        else:
            if letters:
                haraqat.append(current_h)
            letters.append(ch)
            current_h = ""
    if letters:
        haraqat.append(current_h)
    return letters, haraqat


def _haraqat_to_id(haraqat_str: str) -> int:
    """Map a haraqat string (possibly combined like 'َّ') to its vocab ID.

    Returns the index into TARGET_VOCAB (which is [pad] + haraqat keys + extra).
    """
    if haraqat_str in _HARAQAT_TO_ID:
        return _HARAQAT_TO_ID[haraqat_str] + 1  # offset for pad at index 0
    return 0  # unknown haraqat → no-diacritic


class TashkeelaDataset(Dataset):
    """Tashkeela Arabic diacritization dataset.

    Files: train-NNN.txt (sharded), val-NNN.txt, test-NNN.txt — one
    diacritized Arabic line per row. Falls back to {split}.txt for
    legacy non-sharded corpora.

    Shards let us host large corpora on GitHub without LFS (each shard
    stays under the 100MB file-size limit).
    """

    def __init__(self, split: str, root: Path | None = None, cleaner: str = "arabic"):
        self.root = Path(root) if root else DEFAULT_TASHKEELA_ROOT
        if not self._locate_split(split):
            raise FileNotFoundError(f"Tashkeela {split} not found under {self.root}")
        self.examples = self._load(split, cleaner)
        self.split = split

    def _locate_split(self, split: str) -> Path | None:
        """Return the first shard path for a split, or None if missing."""
        # Sharded: train-001.txt, train-002.txt, ...
        shards = sorted(self.root.glob(f"{split}-*.txt"))
        if shards:
            return shards[0]
        # Legacy: single {split}.txt
        legacy = self.root / f"{split}.txt"
        if legacy.is_file():
            return legacy
        return None

    def _iter_split_lines(self, split: str):
        """Yield lines from all shards (or the single legacy file) for a split."""
        shards = sorted(self.root.glob(f"{split}-*.txt"))
        if shards:
            for shard in shards:
                for line in shard.read_text(encoding="utf-8").splitlines():
                    yield line
            return
        legacy = self.root / f"{split}.txt"
        if legacy.is_file():
            for line in legacy.read_text(encoding="utf-8").splitlines():
                yield line

    def _load(self, split: str, cleaner: str) -> list[Example]:
        enc = ArabicEncoder(cleaner=cleaner)
        out: list[Example] = []
        for line in self._iter_split_lines(split):
            line = line.strip()
            if not line:
                continue
            cleaned = enc.clean(line)
            if not cleaned:
                continue
            letters, haraqat = _extract_pairs(cleaned)
            input_ids = enc.encode(strip_haraqat_chars("".join(letters)))
            target_ids = [_haraqat_to_id(h) for h in haraqat]
            n = min(len(input_ids), len(target_ids))
            input_ids = input_ids[:n]
            target_ids = target_ids[:n]
            if n == 0:
                continue
            out.append(Example(input_ids=input_ids, target_ids=target_ids, raw=line))
        return out

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def strip_haraqat_chars(text: str) -> str:
    """Drop haraqat chars from a string."""
    return "".join(c for c in text if c not in _HARAQAT_CHARS)


def load_tashkeela(
    split: str,
    root: Path | None = None,
    cleaner: str = "arabic",
) -> TashkeelaDataset:
    """Convenience loader."""
    return TashkeelaDataset(split=split, root=root, cleaner=cleaner)


# ---- MLM pretraining -------------------------------------------------

@dataclass(frozen=True)
class MLMExample:
    """One MLM training example: masked input + per-position targets.

    `target_ids[i] == PAD_ID` means "don't compute loss at position i"
    (either the position was not selected for masking, or it's padding).
    """
    input_ids: list[int]      # masked input
    target_ids: list[int]     # original IDs at masked positions, PAD elsewhere
    raw: str


def _apply_bert_mask(
    input_ids: list[int],
    mask_prob: float,
    rng: random.Random,
    vocab_size: int,
    mask_id: int,
) -> tuple[list[int], list[int]]:
    """BERT-style masking on a single sequence.

    Selects `mask_prob` of non-PAD positions. Of selected:
      - 80% → mask_id
      - 10% → random token (uniform over [1, vocab_size-1], excluding PAD)
      - 10% → unchanged

    The `mask_id` and `vocab_size` are passed in (not imported) so the
    same function works for any language's vocab.

    Returns (masked_input, target) where target has the original ID at
    selected positions and PAD_ID elsewhere.
    """
    n = len(input_ids)
    masked = list(input_ids)
    target = [PAD_ID] * n
    for i, original in enumerate(input_ids):
        if original == PAD_ID:
            continue
        if rng.random() >= mask_prob:
            continue
        target[i] = original
        r = rng.random()
        if r < 0.8:
            masked[i] = mask_id
        elif r < 0.9:
            masked[i] = rng.randint(1, vocab_size - 1)
        # else: leave unchanged
    return masked, target


class ArabicMLMDataset(Dataset):
    """Undiacritized Arabic text for masked-LM pretraining.

    Loads raw Arabic lines, strips haraqat, encodes to IDs, and applies
    BERT-style masking on the fly in __getitem__ (each epoch sees fresh
    masks).
    """

    def __init__(
        self,
        split: str = "train",
        root: Path | None = None,
        cleaner: str = "arabic",
        mask_prob: float = 0.15,
        max_len: int = 200,
        seed: int = 42,
    ) -> None:
        self.root = Path(root) if root else ARABIC_MLM_ROOT
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.base_seed = seed
        self.vocab_size = INPUT_VOCAB_SIZE  # Arabic
        self.mask_id = MASK_ID  # Arabic
        enc = ArabicEncoder(cleaner=cleaner)
        # Support both sharded (train-001.txt) and legacy (train.txt) layouts.
        shards = sorted(self.root.glob(f"{split}-*.txt"))
        if shards:
            lines: list[str] = []
            for shard in shards:
                lines.extend(shard.read_text(encoding="utf-8").splitlines())
        else:
            path = self.root / f"{split}.txt"
            if not path.is_file():
                raise FileNotFoundError(f"MLM corpus {split} not found at {path}")
            lines = path.read_text(encoding="utf-8").splitlines()
        self.sequences: list[tuple[list[int], str]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned = enc.clean(line)
            if not cleaned:
                continue
            undiacritized = strip_haraqat_chars(cleaned)
            ids = enc.encode(undiacritized)[:max_len]
            if len(ids) < 4:
                continue
            self.sequences.append((ids, line))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> MLMExample:
        ids, raw = self.sequences[idx]
        seed = hash((self.base_seed, idx)) & 0xFFFFFFFF
        rng = random.Random(seed)
        masked, target = _apply_bert_mask(ids, self.mask_prob, rng, self.vocab_size, self.mask_id)
        return MLMExample(input_ids=masked, target_ids=target, raw=raw)


def load_arabic_mlm(
    split: str = "train",
    root: Path | None = None,
    cleaner: str = "arabic",
    mask_prob: float = 0.15,
    max_len: int = 200,
    seed: int = 42,
) -> ArabicMLMDataset:
    """Convenience loader for the MLM dataset."""
    return ArabicMLMDataset(
        split=split, root=root, cleaner=cleaner,
        mask_prob=mask_prob, max_len=max_len, seed=seed,
    )


# ---- Hebrew: Nakdimon multi-target ----------------------------------

# Default corpus location — Modern + Biblical pointed Hebrew.
# Tries the combined path first (data/sefaria + distilled), then individual.
def _build_combined_hebrew_corpus(target: Path) -> Path:
    """Build combined Hebrew corpus from Sefaria + distilled + Nakdimon.

    Writes train/val/test to `target`. The combined corpus has significantly
    more data than raw Nakdimon alone, reducing overfitting for seq2seq.
    """
    target.mkdir(parents=True, exist_ok=True)
    sefaria = Path("/opt/rababa/data/sefaria")
    distilled = Path("/opt/rababa/data/hebrew-distilled")
    nakdimon_vol = Path("/datasets/nakdimon")
    for split in ("train", "val", "test"):
        parts = []
        # Nakdimon (always include if available — gold standard diacritized).
        nak_path = nakdimon_vol / f"{split}.txt"
        if nak_path.is_file():
            parts.append(nak_path.read_text(encoding="utf-8"))
        # Sefaria (Biblical pointed Hebrew).
        for name in (f"{split}.txt", f"sefaria_{split}/{split}.txt"):
            p = sefaria / name
            if p.is_file():
                parts.append(p.read_text(encoding="utf-8"))
                break
        # Distilled (Modern Hebrew, diacritized by Dicta API).
        for name in (f"{split}.txt", f"hebrew_distilled_{split}/{split}.txt"):
            p = distilled / name
            if p.is_file():
                parts.append(p.read_text(encoding="utf-8"))
                break
        (target / f"{split}.txt").write_text("".join(parts), encoding="utf-8")
    return target


def _find_nakdimon_root() -> Path:
    # Check volume-mounted combined corpus first (persistent across containers).
    vol_combined = Path("/datasets/nakdimon-combined")
    if (vol_combined / "train.txt").is_file():
        return vol_combined
    # Check container-local combined corpus (built by fetch_data).
    local_combined = Path("/opt/rababa/data/nakdimon-combined")
    if (local_combined / "train.txt").is_file():
        return local_combined
    # Build combined corpus on-the-fly from image repos + volume.
    sefaria = Path("/opt/rababa/data/sefaria")
    distilled = Path("/opt/rababa/data/hebrew-distilled")
    nakdimon_vol = Path("/datasets/nakdimon")
    if any(p.is_dir() for p in (sefaria, distilled)) and nakdimon_vol.is_dir():
        try:
            return _build_combined_hebrew_corpus(vol_combined)
        except Exception:
            pass  # Fall through to raw Nakdimon.
    # Fall back to raw Nakdimon on volume.
    if (nakdimon_vol / "train.txt").is_file():
        return nakdimon_vol
    # Local dev fallback.
    local = Path(__file__).resolve().parent.parent.parent / "data" / "nakdimon"
    return local


DEFAULT_NAKDIMON_ROOT = _find_nakdimon_root()

_NIQQUD_TO_ID = {n: i for i, n in enumerate(NIQQUD_VOCAB)}
_DAGESH_TO_ID = {d: i for i, d in enumerate(DAGESH_VOCAB)}
_SIN_TO_ID = {s: i for i, s in enumerate(SIN_VOCAB)}
_NIQQUD_SET = set(NIQQUD_VOCAB) - {NIQQUD_VOCAB[0], NIQQUD_VOCAB[1]}  # actual marks (no pad/RAFE)
_SIN_SET = set(SIN_VOCAB) - {SIN_VOCAB[0], SIN_VOCAB[1]}


@dataclass(frozen=True)
class HebrewExample:
    """One parallel pair for Hebrew diacritization.

    All four lists have the same length. `*_target_ids[i] == PAD_ID`
    means "skip this position in the loss for that head" (the letter
    cannot take that mark category).
    """
    input_ids: list[int]
    niqqud_ids: list[int]
    dagesh_ids: list[int]
    sin_ids: list[int]
    raw: str


@dataclass(frozen=True)
class HebrewMLMExample:
    input_ids: list[int]
    target_ids: list[int]
    raw: str


def _iterate_dotted_hebrew(text: str):
    """Port of Nakdimon's iterate_dotted_text.

    Yields (letter, niqqud_char, dagesh_char, sin_char) per Hebrew-letter
    position. Combining-mark order is not assumed — we classify each
    following char as dagesh / sin / niqqud / other and consume until we
    hit a non-mark (next letter, whitespace, or punctuation).
    """
    n = len(text)
    i = 0
    while i < n:
        letter = text[i]
        i += 1
        dagesh = ""
        sin = ""
        niqqud = ""
        if is_hebrew_letter(letter):
            # Consume any subsequent Hebrew combining marks (order-agnostic).
            while i < n:
                c = text[i]
                if c == "ּ":  # DAGESH_LETTER
                    if dagesh == "":
                        dagesh = c
                    i += 1
                elif c in _SIN_SET:
                    if sin == "":
                        sin = c
                    i += 1
                elif c in _NIQQUD_SET:
                    if niqqud == "":
                        niqqud = c
                    i += 1
                else:
                    break
            # Special case: ו + dagesh + no niqqud → treat as SHURUK.
            if letter == "ו" and dagesh == "ּ" and niqqud == "":
                dagesh = ""
                niqqud = "ּ"
        yield letter, niqqud, dagesh, sin


def _hebrew_marks_to_targets(
    letter: str,
    niqqud_char: str,
    dagesh_char: str,
    sin_char: str,
) -> tuple[int, int, int]:
    """Convert (letter, marks) to (niqqud_id, dagesh_id, sin_id) target IDs.

    Positions where the letter can't take a category get PAD_ID (skip in loss).
    Positions that can but don't have a mark get RAFE ID ("decided: none").
    """
    n_id = _NIQQUD_TO_ID.get(niqqud_char) if can_niqqud(letter) else PAD_ID
    if n_id is None:
        n_id = _NIQQUD_TO_ID["ֿ"]  # RAFE
    d_id = _DAGESH_TO_ID.get(dagesh_char) if can_dagesh(letter) else PAD_ID
    if d_id is None:
        d_id = _DAGESH_TO_ID["ֿ"]
    s_id = _SIN_TO_ID.get(sin_char) if can_sin(letter) else PAD_ID
    if s_id is None:
        s_id = _SIN_TO_ID["ֿ"]
    return n_id, d_id, s_id


class NakdimonDataset(Dataset):
    """Hebrew diacritization dataset — multi-head targets (niqqud/dagesh/sin).

    File format: one fully-pointed Hebrew line per row.
    """

    def __init__(
        self,
        split: str,
        root: Path | None = None,
        cleaner: str = "hebrew",
        max_len: int = 200,
    ) -> None:
        self.root = Path(root) if root else DEFAULT_NAKDIMON_ROOT
        path = self.root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Nakdimon {split} not found at {path}")
        self.split = split
        self.max_len = max_len
        self.examples = self._load(path, cleaner)

    def _load(self, path: Path, cleaner: str) -> list[HebrewExample]:
        enc = HebrewEncoder(cleaner=cleaner)
        out: list[HebrewExample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            input_ids: list[int] = []
            niqqud_ids: list[int] = []
            dagesh_ids: list[int] = []
            sin_ids: list[int] = []
            for letter, nch, dch, sch in _iterate_dotted_hebrew(line):
                # Normalize letter the same way the encoder will at inference time.
                cleaned = enc.clean(letter)
                if not cleaned:
                    continue
                ids = enc.encode(cleaned)
                if not ids:
                    continue
                # Take only the first encoded ID (normalized letter → single token).
                input_ids.append(ids[0])
                n_id, d_id, s_id = _hebrew_marks_to_targets(letter, nch, dch, sch)
                niqqud_ids.append(n_id)
                dagesh_ids.append(d_id)
                sin_ids.append(s_id)
                if len(input_ids) >= self.max_len:
                    break
            if len(input_ids) < 4:
                continue
            out.append(HebrewExample(
                input_ids=input_ids,
                niqqud_ids=niqqud_ids,
                dagesh_ids=dagesh_ids,
                sin_ids=sin_ids,
                raw=line,
            ))
        return out

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> HebrewExample:
        return self.examples[idx]


def load_nakdimon(
    split: str,
    root: Path | None = None,
    cleaner: str = "hebrew",
    max_len: int = 200,
) -> NakdimonDataset:
    return NakdimonDataset(split=split, root=root, cleaner=cleaner, max_len=max_len)


class HebrewMLMDataset(Dataset):
    """Undiacritized Hebrew text for MLM pretraining (parallel to ArabicMLMDataset)."""

    def __init__(
        self,
        split: str = "train",
        root: Path | None = None,
        cleaner: str = "hebrew",
        mask_prob: float = 0.15,
        max_len: int = 200,
        seed: int = 42,
    ) -> None:
        self.root = Path(root) if root else DEFAULT_NAKDIMON_ROOT
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.base_seed = seed
        self.vocab_size = HEBREW_INPUT_VOCAB_SIZE
        self.mask_id = HEBREW_MASK_ID
        enc = HebrewEncoder(cleaner=cleaner)
        path = self.root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Hebrew MLM corpus {split} not found at {path}")
        self.sequences: list[tuple[list[int], str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip diacritics by re-iterating dotted text and keeping only letters.
            letters = [letter for letter, *_ in _iterate_dotted_hebrew(line)]
            cleaned = enc.clean("".join(letters))
            ids = enc.encode(cleaned)[:max_len]
            if len(ids) < 4:
                continue
            self.sequences.append((ids, line))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> HebrewMLMExample:
        ids, raw = self.sequences[idx]
        seed = hash((self.base_seed, idx)) & 0xFFFFFFFF
        rng = random.Random(seed)
        masked, target = _apply_bert_mask(ids, self.mask_prob, rng, self.vocab_size, self.mask_id)
        return HebrewMLMExample(input_ids=masked, target_ids=target, raw=raw)


def load_hebrew_mlm(
    split: str = "train",
    root: Path | None = None,
    cleaner: str = "hebrew",
    mask_prob: float = 0.15,
    max_len: int = 200,
    seed: int = 42,
) -> HebrewMLMDataset:
    return HebrewMLMDataset(
        split=split, root=root, cleaner=cleaner,
        mask_prob=mask_prob, max_len=max_len, seed=seed,
    )

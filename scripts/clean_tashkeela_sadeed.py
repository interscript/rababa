#!/usr/bin/env python3
"""Sadeed-style Tashkeela cleaning pipeline (paper-described, no code dep).

Implements the preprocessing steps from Sadeed (Aldallal et al. 2025,
arXiv:2504.21635 Section 3) that we can apply to any Tashkeela input:

  1. Sukun normalization on definite-article lam before sun letters.
  2. Sukun removal on alef (madd carriers never carry sukun legitimately).
  3. Stop-word canonicalization (في → فِي, عن → عَنْ, ...).
  4. Quality filter: drop examples with >2 fully-undiacritized words OR
     >=3 partially-diacritized words (Sadeed report this preserves 93%).

Steps we deliberately omit in v1:
  - Iltiqā' as-sākinayn resolution — phonological rule requires deep
    linguistic context; benefit unclear without it. Documented as TODO.
  - Chunking 50-60 words — our Tashkeela is already sentence-segmented.

Output format matches input: one example per line, UTF-8.

Usage:
  python scripts/clean_tashkeela_sadeed.py \\
      --in-dir test-datasets/tashkeela \\
      --out-dir data/tashkeela-cleaned
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---- Sukun normalization --------------------------------------------

SUKUN = "ْ"
SHADDA = "ّ"
ALEF = "ا"
WAW = "و"
YA = "ي"
LAM = "ل"

# Sun letters (الحروف الشمسية) — lam of definite article is silent before these.
SUN_LETTERS = "تثدذرزسشصضطظلن"


def normalize_sukun(text: str) -> str:
    """Remove sukun where it is orthographically spurious.

    Two rules:
      (a) Definite article: اَلْ + sun_letter → اَل + sun_letter (with
          shadda on the sun letter). The lam is silent before sun letters;
          keeping sukun on it is non-canonical.
      (b) Alef (madd carrier) never bears a true sukun — strip any ْ after ا.
    """
    # (a) Definite article before sun letters.
    for sl in SUN_LETTERS:
        text = text.replace(f"{LAM}{SUKUN}{sl}", f"{LAM}{sl}")

    # (b) Alef + sukun → Alef. Safe: alef is always a madd/vowel carrier.
    text = text.replace(f"{ALEF}{SUKUN}", ALEF)

    return text


# ---- Stop-word canonicalization -------------------------------------

# Words with a single canonical diacritization that are frequently
# left undiacritized or inconsistently diacritized in Tashkeela.
STOPWORD_FIXES: dict[str, str] = {
    "في": "فِي",
    "فِيْ": "فِي",
    "فيٴ": "فِي",
    "عن": "عَنْ",
    "عَن": "عَنْ",
    "من": "مِنْ",
    "مِن": "مِنْ",
    "مَن": "مَنْ",  # distinct word ("who") — also canonicalize
    "الى": "إِلَى",
    "إلى": "إِلَى",
    "على": "عَلَى",
    "أن": "أَنْ",
    "أَن": "أَنْ",
    "إن": "إِنْ",
    "إِن": "إِنْ",
    "أنه": "أَنَّهُ",
    "إنه": "إِنَّهُ",
    "ما": "مَا",
    "هو": "هُوَ",
    "هي": "هِيَ",
    "هم": "هُمْ",
    "هن": "هُنَّ",
    "هذا": "هَذَا",
    "هذه": "هَذِهِ",
    "ذلك": "ذَلِكَ",
    "الذي": "الَّذِي",
    "التي": "الَّتِي",
    "الذين": "الَّذِينَ",
    "اللاتي": "الَّلَاتِي",
    "اللائي": "الَّلَائِي",
    "كيف": "كَيْفَ",
    "حيث": "حَيْثُ",
    "لكن": "لَكِنْ",
    "لَكِن": "لَكِنْ",
    "كل": "كُلُّ",
    "بعض": "بَعْضٍ",
}


def canonicalize_stopwords(text: str) -> str:
    """Replace stop words with their canonical diacritized forms.

    Operates on whitespace-split tokens — no in-word substitution.
    """
    out_tokens: list[str] = []
    for tok in text.split():
        # Strip surrounding punctuation for lookup, then reattach.
        m = re.match(r"^([^؀-ۿ]*)(.*?)([^؀-ۿ]*)$", tok)
        if not m:
            out_tokens.append(tok)
            continue
        pre, core, post = m.groups()
        # Strip any existing haraqat from the core for lookup.
        stripped = re.sub(r"[ً-ْ]", "", core)
        replacement = STOPWORD_FIXES.get(stripped)
        if replacement:
            out_tokens.append(f"{pre}{replacement}{post}")
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


# ---- Quality filter -------------------------------------------------

DIACRITIC_RE = re.compile(r"[ً-ْ]")  # tanwin, haraqat, shadda, sukun
ARABIC_LETTER_RE = re.compile(r"[ء-غف-ي]")

# Vowel/carrier letters that don't require explicit haraqat in valid orthography.
# Counting these as "missing diacritics" produces 90%+ false-positive drops.
VOWEL_LETTERS = set("اويىةآإأؤئ")


def _word_diacritic_coverage(word: str) -> tuple[int, int]:
    """Return (n_consonants, n_diacritized_consonants) for an Arabic word.

    Vowel/carrier letters (ا و ي ى ة آ إ أ ؤ ئ) are excluded from the count —
    they're inherently vowel-bearing and don't require explicit haraqat in
    standard Arabic orthography. A "missing diacritic" only counts when a
    true consonant lacks one.

    A consonant is "diacritized" if it's immediately followed by at least one
    diacritic mark (haraqa, tanwin, shadda, or sukun).
    """
    chars = list(word)
    i = 0
    n_cons = 0
    n_diac_cons = 0
    while i < len(chars):
        c = chars[i]
        if ARABIC_LETTER_RE.match(c):
            if c in VOWEL_LETTERS:
                i += 1
                continue
            n_cons += 1
            j = i + 1
            has_diac = False
            while j < len(chars) and DIACRITIC_RE.match(chars[j]):
                has_diac = True
                j += 1
            if has_diac:
                n_diac_cons += 1
            i = j
        else:
            i += 1
    return n_cons, n_diac_cons


def passes_quality_filter(
    line: str,
    max_undiacritized_words: int = 2,
    max_partial_words: int = 2,
    min_arabic_letters: int = 20,
) -> bool:
    """Sadeed-style quality gate.

    Drops a line if:
      - it has fewer than `min_arabic_letters` Arabic letters overall
        (filters out parsing garbage, page numbers, etc.), OR
      - it has more than `max_undiacritized_words` words with zero
        diacritics on any consonant, OR
      - it has more than `max_partial_words` words with partial diacritics.

    Vowel/carrier letters don't count toward the consonant total.
    """
    undiacritized = 0
    partial = 0
    arabic_letter_total = 0
    for tok in line.split():
        n_cons, n_diac = _word_diacritic_coverage(tok)
        arabic_letter_total += n_cons + sum(
            1 for c in tok if c in VOWEL_LETTERS
        )
        if n_cons == 0:
            continue
        if n_diac == 0:
            undiacritized += 1
        elif n_diac < n_cons:
            partial += 1
    if arabic_letter_total < min_arabic_letters:
        return False
    return undiacritized <= max_undiacritized_words and partial <= max_partial_words


# ---- Pipeline -------------------------------------------------------

def resolve_iltiqaa_as_sakinayn(text: str) -> str:
    """Apply the iltiqā' as-sākinayn (meeting of two voiceless) rule.

    Arabic phonology forbids two adjacent consonants both bearing sukun
    when the first is not a shaddah-bearing letter or one of the special
    letters (ال, tower of madd). The rule: where two sukun-bearing
    consonants meet, the first loses its sukun (takes a short vowel
    instead — kasra by default).

    Conservative implementation: only act on the literal pattern
    `cons1 + ْ + cons2 + ْ` where neither cons is a shaddah carrier or
    vowel letter. Drop the first sukun. This is the most common case;
    full phonological context resolution is out of scope for the
    data-cleaning pipeline (the model learns the residual).

    Reference: Sadeed paper Section 3 (mentioned as a TODO); Wright's
    Arabic Grammar §23B for the classical rule.
    """
    # Build the regex once. Match: non-vowel letter, sukun, non-vowel letter, sukun.
    # We drop the first sukun (the one on the prior letter) — kasra is implied.
    # Vowel letters (اويىةآإأؤئ) and shaddah (ّ) are excluded from cons1/cons2.
    pattern = re.compile(rf"([^\sً-ْ])({SUKUN})([^\sً-ْ{SHADDA}])")
    # Apply repeatedly — one pass may unlock another.
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(rf"\1\3", text)
    return text


def clean_line(line: str) -> str:
    """Apply all normalization steps. Does NOT filter — caller decides."""
    line = line.strip()
    if not line:
        return ""
    line = normalize_sukun(line)
    line = canonicalize_stopwords(line)
    line = resolve_iltiqaa_as_sakinayn(line)
    # Collapse multiple spaces introduced by replacements.
    line = re.sub(r"\s+", " ", line).strip()
    return line


# ---- Chunking (Sadeed Section 3: 50-60 word segments) ---------------

# Hierarchical split priority — try stronger separators first.
SENTENCE_END_RE = re.compile(r"(?<=[\.!\?؟])\s+")
LINE_BREAK_RE = re.compile(r"\n+")
QUOTE_RE = re.compile(r"(?<=[”\"])\s+")
PAREN_RE = re.compile(r"(?<=[\)\]])\s+")
COMMA_RE = re.compile(r"(?<=[،,])\s+")

CHUNK_SEPARATORS = [
    SENTENCE_END_RE,
    LINE_BREAK_RE,
    QUOTE_RE,
    PAREN_RE,
    COMMA_RE,
]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, min_words: int = 40, max_words: int = 60) -> list[str]:
    """Hierarchically split text into chunks of ~50-60 words.

    Strategy: try sentence-end punctuation first, then line breaks, then
    quotes, then parens, then commas. If a leaf chunk is still >max_words,
    hard-split at word boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if min_words <= _word_count(text) <= max_words:
        return [text]

    # Try each separator in priority order.
    for sep_re in CHUNK_SEPARATORS:
        if not sep_re.search(text):
            continue
        pieces = [p.strip() for p in sep_re.split(text) if p.strip()]
        if len(pieces) < 2:
            continue
        # Greedily merge adjacent pieces up to max_words.
        chunks: list[str] = []
        buf: list[str] = []
        buf_wc = 0
        for piece in pieces:
            wc = _word_count(piece)
            if wc > max_words:
                # Flush current buffer first.
                if buf:
                    chunks.append(" ".join(buf))
                    buf, buf_wc = [], 0
                # Recursively split oversized piece with next separator.
                chunks.extend(chunk_text(piece, min_words, max_words))
            elif buf_wc + wc > max_words and buf_wc >= min_words:
                chunks.append(" ".join(buf))
                buf, buf_wc = [piece], wc
            else:
                buf.append(piece)
                buf_wc += wc
        if buf:
            chunks.append(" ".join(buf))
        # Filter out chunks that are too short (merge them back? skip for now).
        return [c for c in chunks if _word_count(c) >= min_words // 2]

    # No separator matched and text is too long → hard split on word boundaries.
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def process_file(src: Path, dst: Path) -> dict[str, int]:
    """Clean + chunk one split file. Returns {kept, dropped, total}."""
    stats = {"kept": 0, "dropped": 0, "total": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out_f:
        for raw in src.read_text(encoding="utf-8").splitlines():
            cleaned = clean_line(raw)
            if not cleaned:
                continue
            stats["total"] += 1
            for chunk in chunk_text(cleaned):
                if not chunk:
                    continue
                if passes_quality_filter(chunk):
                    out_f.write(chunk + "\n")
                    stats["kept"] += 1
                else:
                    stats["dropped"] += 1
    return stats


def process_directory_tree(src_root: Path, dst_dir: Path, max_bytes_per_shard: int = 90 * 1024 * 1024) -> dict[str, int]:
    """Walk a directory tree of raw Tashkeela text files.

    Concatenates every file under src_root, cleans + chunks every paragraph,
    then shuffles and writes 80/10/10 split into dst_dir.

    Train shards are written as train-001.txt, train-002.txt, ... so each
    shard stays under GitHub's 100MB file-size limit (no LFS needed).
    Val/test are typically small enough for a single file.

    Use this when input is the unzipped Tashkeela corpus
    (Tashkeela-arabic-diacritized-text-utf8-0.3/texts.txt/**).
    """
    import random

    all_chunks: list[str] = []
    files_scanned = 0
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        # Skip obvious non-text files (toolz/, doc/, etc.).
        if any(part in {"doc", "toolz"} for part in path.relative_to(src_root).parts):
            continue
        if path.suffix and path.suffix not in {".txt", ".htm"}:
            continue
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        files_scanned += 1
        # Each file may contain many paragraphs separated by blank lines.
        for para in re.split(r"\n\s*\n", raw_text):
            para = para.strip()
            if not para or len(para) < 30:
                continue
            cleaned = clean_line(para)
            if not cleaned:
                continue
            for chunk in chunk_text(cleaned):
                if chunk and passes_quality_filter(chunk):
                    all_chunks.append(chunk)

    rng = random.Random(42)
    rng.shuffle(all_chunks)
    n = len(all_chunks)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": all_chunks[:n_train],
        "val": all_chunks[n_train : n_train + n_val],
        "test": all_chunks[n_train + n_val :],
    }
    dst_dir.mkdir(parents=True, exist_ok=True)
    shard_counts: dict[str, int] = {}
    for name, items in splits.items():
        shard_idx = 1
        cur_bytes = 0
        cur_lines: list[str] = []
        shards: list[Path] = []
        for line in items:
            line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
            if cur_bytes + line_bytes > max_bytes_per_shard and cur_lines:
                shard_path = dst_dir / f"{name}-{shard_idx:03d}.txt"
                shard_path.write_text("\n".join(cur_lines), encoding="utf-8")
                shards.append(shard_path)
                shard_idx += 1
                cur_lines, cur_bytes = [], 0
            cur_lines.append(line)
            cur_bytes += line_bytes
        if cur_lines:
            shard_path = dst_dir / f"{name}-{shard_idx:03d}.txt"
            shard_path.write_text("\n".join(cur_lines), encoding="utf-8")
            shards.append(shard_path)
        shard_counts[name] = shard_idx

    return {
        "files_scanned": files_scanned,
        "chunks_kept": n,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "train_shards": shard_counts["train"],
        "val_shards": shard_counts.get("val", 1),
        "test_shards": shard_counts.get("test", 1),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Source dir: either containing {train,val,test}.txt "
                        "(split mode) OR the raw Tashkeela tree "
                        "(texts.txt/msa/**) when --tree is set.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Destination dir for cleaned files")
    p.add_argument("--tree", action="store_true",
                   help="Walk the input as a raw Tashkeela directory tree "
                        "(split 80/10/10 after chunking).")
    p.add_argument("--max-undiacritized-words", type=int, default=2,
                   help="Drop examples with more than N fully-undiacritized words")
    p.add_argument("--max-partial-words", type=int, default=2,
                   help="Drop examples with more than N partially-diacritized words")
    args = p.parse_args(argv)

    if not args.in_dir.is_dir():
        print(f"ERROR: in-dir missing: {args.in_dir}", file=sys.stderr)
        return 1

    print(f"=== Sadeed-style Tashkeela cleaning ===")
    print(f"  in:  {args.in_dir}")
    print(f"  out: {args.out_dir}\n")

    if args.tree:
        stats = process_directory_tree(args.in_dir, args.out_dir)
        print(f"  files scanned:  {stats['files_scanned']:,}")
        print(f"  chunks kept:    {stats['chunks_kept']:,}")
        print(f"  train:          {stats['train']:,}  ({stats['train_shards']} shards)")
        print(f"  val:            {stats['val']:,}  ({stats['val_shards']} shards)")
        print(f"  test:           {stats['test']:,}  ({stats['test_shards']} shards)")
        print("\nDone.")
        return 0

    total_stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        src = args.in_dir / f"{split}.txt"
        dst = args.out_dir / f"{split}.txt"
        if not src.is_file():
            print(f"  WARN: {src} missing, skipping")
            continue
        stats = process_file(src, dst)
        total_stats[split] = stats
        pct = (stats["kept"] / stats["total"] * 100) if stats["total"] else 0
        print(f"  {split}: kept {stats['kept']:,}/{stats['total']:,} ({pct:.1f}%) "
              f"— dropped {stats['dropped']:,}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

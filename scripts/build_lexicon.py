#!/usr/bin/env python3
"""Build the word-level haraqat lexicon from training data.

Reads {train,val}-*.txt files (one diacritized Arabic line per row),
splits each line into (letters, haraqat) pairs, groups by word, and
writes a JSON lexicon suitable for trie-constrained decoding.

Output: lexicon.json with shape
    {"undiacritized_word": [[haraqat_id_per_char, ...], ...]}

Each entry is pruned to top-K most frequent haraqat sequences (default
K=5). Words seen fewer than `--min-word-freq` times (default 2) are
dropped — they're statistically unreliable.

Usage:
    python scripts/build_lexicon.py \\
        --data-dir data/tashkeela-full \\
        --output  models/rababa_arabic_pro/lexicon.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rababa.constants import ALL_POSSIBLE_HARAQAT, VALID_ARABIC  # noqa: E402
from rababa.decoding.lexicon import Lexicon, save_lexicon  # noqa: E402
from rababa.datasets import _extract_pairs, strip_haraqat_chars  # noqa: E402
from rababa.encoder import ArabicEncoder  # noqa: E402


HARAQAT_SET = set(ALL_POSSIBLE_HARAQAT.keys())
HARAQAT_TO_ID = {h: i for i, h in enumerate(ALL_POSSIBLE_HARAQAT.keys())}
_ID_TO_CHAR = [""] + VALID_ARABIC  # PAD at 0


def _haraqat_to_ids(haraqat_str: str) -> int:
    """Same mapping used in datasets.TashkeelaDataset."""
    if haraqat_str in HARAQAT_TO_ID:
        return HARAQAT_TO_ID[haraqat_str] + 1  # offset for pad at index 0
    return 0


def iter_train_lines(data_dir: Path, split: str = "train"):
    """Yield raw diacritized lines from {split}-*.txt or {split}.txt."""
    shards = sorted(data_dir.glob(f"{split}-*.txt"))
    if shards:
        for shard in shards:
            for line in shard.read_text(encoding="utf-8").splitlines():
                yield line
        return
    legacy = data_dir / f"{split}.txt"
    if legacy.is_file():
        for line in legacy.read_text(encoding="utf-8").splitlines():
            yield line


def words_from_line(line: str, enc: ArabicEncoder) -> list[tuple[str, tuple[int, ...]]]:
    """Return list of (undiacritized_word, haraqat_ids) for one diacritized line."""
    cleaned = enc.clean(line)
    if not cleaned:
        return []
    letters, haraqat = _extract_pairs(cleaned)
    # Group letters into words by whitespace.
    words: list[tuple[str, tuple[int, ...]]] = []
    cur_letters: list[str] = []
    cur_haraqat: list[int] = []
    for ch, h in zip(letters, haraqat):
        if ch == " ":
            if cur_letters:
                undiac = strip_haraqat_chars("".join(cur_letters))
                words.append((undiac, tuple(cur_haraqat)))
            cur_letters = []
            cur_haraqat = []
        else:
            cur_letters.append(ch)
            cur_haraqat.append(_haraqat_to_ids(h))
    if cur_letters:
        undiac = strip_haraqat_chars("".join(cur_letters))
        words.append((undiac, tuple(cur_haraqat)))
    return words


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dir containing train-*.txt (or train.txt)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output lexicon JSON path")
    p.add_argument("--splits", nargs="+", default=["train"],
                   help="Splits to scan (default: train only)")
    p.add_argument("--top-k-per-word", type=int, default=5,
                   help="Keep only top-K most frequent haraqat sequences per word")
    p.add_argument("--min-word-freq", type=int, default=2,
                   help="Drop words seen fewer than this many times")
    args = p.parse_args(argv)

    enc = ArabicEncoder(cleaner="arabic")
    lex = Lexicon(top_k_per_word=args.top_k_per_word,
                  min_word_freq=args.min_word_freq)
    n_lines = 0
    n_words = 0
    for split in args.splits:
        for line in iter_train_lines(args.data_dir, split):
            line = line.strip()
            if not line:
                continue
            for word, ids in words_from_line(line, enc):
                if word and len(ids) > 0:
                    lex.add(word, ids)
                    n_words += 1
            n_lines += 1

    stats = save_lexicon(lex, args.output)
    print(f"Processed {n_lines:,} lines, {n_words:,} word occurrences")
    print(f"Lexicon: {stats['entries']:,} entries, {stats['sequences']:,} sequences, "
          f"{stats['mb']} MB → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

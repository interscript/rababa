#!/usr/bin/env python3
"""Fetch a Wikipedia language corpus via HuggingFace datasets.

Outputs train/val/test.txt (80/10/10 split) suitable for MLM pretrain
or as a source for downstream distillation.

Usage:
  python scripts/fetch_wiki_corpus.py --lang ar --out data/arwiki --max-lines 500000
  python scripts/fetch_wiki_corpus.py --lang he --out data/hewiki --max-lines 100000
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path


def clean_wiki_text(text: str) -> str:
    """Strip templates, markup, templates, refs. Keep prose."""
    # Drop templates {{...}} (recursive)
    while "{{" in text and "}}" in text:
        new = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if new == text:
            break
        text = new
    # Drop tables {| ... |}
    text = re.sub(r"\{\|[^}]*\|\}", "", text, flags=re.DOTALL)
    # Drop ref tags and HTML
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    # Drop wiki markup: '''bold''', ''italic'', [[link|text]], [http://...]
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\[\[[^\]]*\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    # Drop headings == ... ==
    text = re.sub(r"^=+.+$", "", text, flags=re.MULTILINE)
    # Drop lists / bullets
    text = re.sub(r"^\*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lang", required=True, help="Wikipedia language code (ar, he, en, ...)")
    p.add_argument("--out", type=Path, required=True, help="Output dir")
    p.add_argument("--max-lines", type=int, default=500_000)
    p.add_argument("--min-line-len", type=int, default=40,
                   help="Skip lines shorter than this (filter nav fragments)")
    p.add_argument("--max-line-len", type=int, default=500,
                   help="Split long paragraphs into chunks ≤ this length")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading wikimedia/wikipedia ({args.lang}) via HF datasets...", flush=True)
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{args.lang}",
        split="train",
        streaming=True,
    )

    rng = random.Random(args.seed)
    lines: list[str] = []

    for i, ex in enumerate(ds):
        if len(lines) >= args.max_lines:
            break
        text = clean_wiki_text(ex.get("text", ""))
        if not text:
            continue
        # Split into sentence-ish chunks on common punctuation.
        for chunk in re.split(r"(?<=[.!?。])\s+|\n+", text):
            chunk = chunk.strip()
            if args.min_line_len <= len(chunk) <= args.max_line_len:
                lines.append(chunk)
                if len(lines) >= args.max_lines:
                    break
        if i % 1000 == 0:
            print(f"  scanned {i:,} articles, kept {len(lines):,} lines", flush=True)

    print(f"\nTotal kept: {len(lines):,} lines")

    rng.shuffle(lines)
    n = len(lines)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": lines[:n_train],
        "val": lines[n_train : n_train + n_val],
        "test": lines[n_train + n_val :],
    }
    for name, items in splits.items():
        path = args.out / f"{name}.txt"
        path.write_text("\n".join(items), encoding="utf-8")
        print(f"  {name}.txt: {len(items):,} lines → {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

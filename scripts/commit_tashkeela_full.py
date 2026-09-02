#!/usr/bin/env python3
"""Commit the Sadeed-style cleaned FULL Tashkeela corpus to its own data repo.

Input: data/tashkeela-full/{train,val,test}.txt produced by
scripts/clean_tashkeela_sadeed.py --tree from the full Tashkeela corpus
(both avcorpus.tar.bz2 + Tashkeela-arabic-diacritized-text-utf8-0.3.zip).

Creates / updates the GitHub repo interscript/rababa-tashkeela-full.

Usage:
  python scripts/commit_tashkeela_full.py --data-dir data/tashkeela-full
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
REPO_NAME = "interscript/rababa-tashkeela-full"
REPO_DIR = Path(__file__).resolve().parent.parent / ".tashkeela-full-repo"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def repo_exists(name: str) -> bool:
    return subprocess.run(
        ["gh", "repo", "view", name], capture_output=True, text=True
    ).returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dir containing train.txt, val.txt, test.txt")
    p.add_argument("--repo-name", default=REPO_NAME,
                   help="Override OWNER/NAME")
    args = p.parse_args(argv)

    if not args.data_dir.is_dir() or not list(args.data_dir.glob("train-*.txt")):
        print(f"ERROR: {args.data_dir}/train-*.txt missing.", file=sys.stderr)
        return 1

    print(f"=== Committing cleaned FULL Tashkeela corpus ===\n")

    # Create or clone the repo.
    if repo_exists(args.repo_name):
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        run(["gh", "repo", "clone", args.repo_name, str(REPO_DIR)])
    else:
        parent = REPO_DIR.parent
        parent.mkdir(parents=True, exist_ok=True)
        run([
            "gh", "repo", "create", args.repo_name, "--public",
            "--description",
            "Full Tashkeela corpus (75M words) with Sadeed-style cleaning "
            "for rababa ML training",
            "--clone",
        ], cwd=parent)
        cloned = parent / args.repo_name.split("/")[-1]
        if cloned.exists() and not REPO_DIR.exists():
            cloned.rename(REPO_DIR)

    # Lay out data dir: tashkeela_full_train/train-NNN.txt etc.
    # Files come in shards (train-001.txt, train-002.txt, ...) to stay under
    # GitHub's 100MB file-size limit without LFS.
    subdirs = {"train": "tashkeela_full_train",
               "val":   "tashkeela_full_val",
               "test":  "tashkeela_full_test"}
    stats: dict[str, int] = {}
    for split, subdir in subdirs.items():
        src_files = sorted((args.data_dir).glob(f"{split}-*.txt"))
        if not src_files:
            src_files = sorted((args.data_dir).glob(f"{split}.txt"))
        if not src_files:
            print(f"ERROR: no {split} files in {args.data_dir}", file=sys.stderr)
            return 1
        dst_dir = REPO_DIR / subdir
        dst_dir.mkdir(exist_ok=True)
        total_lines = 0
        for src in src_files:
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            with dst.open(encoding="utf-8") as f:
                total_lines += sum(1 for _ in f)
        stats[split] = total_lines

    total = sum(stats.values())

    # Count shards per split for README.
    shard_counts = {}
    for split, subdir in subdirs.items():
        shard_counts[split] = len(list((REPO_DIR / subdir).glob(f"{split}-*.txt")))

    # Write README.
    (REPO_DIR / "README.adoc").write_text(f"""= Full Tashkeela corpus (Sadeed-style cleaned) for rababa

== Purpose

The canonical Arabic diacritization training corpus for rababa.
Derived from the **full** Tashkeela corpus (75M words, ~500K cleaned
chunks) — replaces the smaller "Tashkeela processed" subset
(50K sentences) used in rababa v0.x.

== Source

Two archives from the original
https://sourceforge.net/projects/tashkeela/[Tashkeela project on
SourceForge]:

  1. `avcorpus.tar.bz2` — Classical Arabic books from the Shamela
     Library (84 books, ~74M words). The bulk of the corpus.
  2. `Tashkeela-arabic-diacritized-text-utf8-0.3.zip` — Modern Standard
     Arabic subset (389 source files).

Fetched by `scripts/fetch_tashkeela_full.py` in the main
https://github.com/interscript/rababa[rababa] repo.

== Cleaning pipeline

The cleaning pipeline is a from-scratch reimplementation of the
preprocessing described in Section 3 of Aldallal et al. (2025),
"Sadeed: Advancing Arabic Diacritization Through Small Language
Model" (arXiv:2504.21635). **No code dependencies on Sadeed or any
other project.**

Steps:

  1. **Sukun normalization** — drop sukun on definite-article lam before
     sun letters; drop sukun on alef (madd carriers never bear sukun).
  2. **Stopword canonicalization** — replace frequently-ambiguous words
     (في, عن, من, ...) with their canonical diacritized forms.
  3. **Chunking** — hierarchically split long passages into ~50-60 word
     chunks (sentence-end punctuation > line breaks > quotes > parens >
     commas).
  4. **Quality filter** — drop chunks with >2 fully-undiacritized words
     or >2 partially-diacritized words. Also drop chunks with <20 Arabic
     letters (page numbers, parsing artifacts).
  5. **Split** — deterministic 80/10/10 train/val/test (seed=42).

The iltiqā' as-sākinayn phonological rule from Sadeed's pipeline is
**not yet implemented** — scheduled for a follow-up commit.

Implemented in `scripts/clean_tashkeela_sadeed.py` in the main rababa
repo.

== License

This corpus is © Taha Zerrouki and contributors, licensed under
https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html[GPL v2]
as required by the upstream Tashkeela dataset's license. This
redistribution carries the same license.

== Attribution

=== Original Tashkeela corpus

* Author: Taha Zerrouki & Amar Balla
* Paper: "Tashkeela: Novel corpus of Arabic vocalized texts, data for
  auto-diacritization systems", Data in Brief (2017).
  DOI: 10.1016/j.dib.2017.01.011
* Project: https://sourceforge.net/projects/tashkeela/[Tashkeela on
  SourceForge]
* License: GPL v2

=== Cleaning pipeline reference

* Authors: Zeina Aldallal, Sara Chrouf, Khalil Hennara, Mohamed Motaism
  Hamed, Muhammad Hreden, Safwan AlModhayan
* Paper: "Sadeed: Advancing Arabic Diacritization Through Small
  Language Model", arXiv:2504.21635 (2025).
* Note: This corpus is **not** the Sadeed_Tashkeela dataset itself
  (which is hosted gated on HuggingFace). It is an independent
  reimplementation of the same paper-described pipeline applied to the
  same upstream Tashkeela source.

== Stats

Total chunks: {total:,}
Total words: ~27M

[cols="1,>1,>1", options="header"]
|===
| Split | Chunks | Shards
| train | {stats.get('train', 0):,} | {shard_counts.get('train', 1)}
| val | {stats.get('val', 0):,} | {shard_counts.get('val', 1)}
| test | {stats.get('test', 0):,} | {shard_counts.get('test', 1)}
|===

== Layout

Files are sharded to stay under GitHub's 100MB file-size limit (no Git
LFS required).

----
tashkeela_full_train/train-001.txt
tashkeela_full_train/train-002.txt
tashkeela_full_train/train-003.txt
tashkeela_full_val/val-001.txt
tashkeela_full_test/test-001.txt
----

Loaders should glob `{split}-*.txt` and concatenate shards in lexical
order (already sorted by NNN suffix).
""", encoding="utf-8")

    # Stage explicit paths only.
    stage_paths = ["README.adoc"]
    for split, subdir in subdirs.items():
        for shard in sorted((REPO_DIR / subdir).glob(f"{split}-*.txt")):
            stage_paths.append(f"{subdir}/{shard.name}")
    run(["git", "-C", str(REPO_DIR), "add", *stage_paths])
    run(["git", "-C", str(REPO_DIR), "status", "--short"])

    # Check existing commits.
    has_commits = (
        subprocess.run(["git", "-C", str(REPO_DIR), "log", "--oneline"],
                       capture_output=True).returncode == 0
    )

    if has_commits:
        branch = "update-corpus"
        run(["git", "-C", str(REPO_DIR), "checkout", "-b", branch])
        run(["git", "-C", str(REPO_DIR), "commit",
             "-m", "Update cleaned Tashkeela corpus"])
        run(["git", "-C", str(REPO_DIR), "push", "-u", "origin", branch])
        run(["gh", "pr", "create", "--repo", args.repo_name,
             "--title", "Update cleaned Tashkeela corpus",
             "--body", "Automated update of train/val/test splits.",
             "--head", branch])
    else:
        run(["git", "-C", str(REPO_DIR), "commit",
             "-m", f"Initial cleaned Tashkeela corpus ({total:,} chunks)"])
        run(["git", "-C", str(REPO_DIR), "push", "-u", "origin", "main"])

    print(f"\n✓ {args.repo_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

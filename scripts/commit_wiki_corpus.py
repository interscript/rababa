#!/usr/bin/env python3
"""Commit a per-language Wikipedia corpus to its own data repo.

Generic version of commit_sefaria_corpus.py — works for any language.

Usage:
  python scripts/commit_wiki_corpus.py --lang ar --data-dir data/arwiki
  python scripts/commit_wiki_corpus.py --lang he --data-dir data/hewiki
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def repo_exists(name: str) -> bool:
    return subprocess.run(
        ["gh", "repo", "view", name], capture_output=True, text=True
    ).returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lang", required=True, help="Wikipedia language code (ar, he, ...)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--repo-name", default=None,
                   help="Full OWNER/NAME; default interscript/rababa-<lang>wiki")
    args = p.parse_args(argv)

    repo_name = args.repo_name or f"interscript/rababa-{args.lang}wiki"
    repo_dir = Path(__file__).resolve().parent.parent / f".{args.lang}wiki-repo"

    if not args.data_dir.is_dir() or not (args.data_dir / "train.txt").is_file():
        print(f"ERROR: {args.data_dir}/train.txt missing.", file=sys.stderr)
        return 1

    print(f"=== Committing {args.lang} Wikipedia corpus ===\n")

    # Create or clone the repo.
    if repo_exists(repo_name):
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        run(["gh", "repo", "clone", repo_name, str(repo_dir)])
    else:
        parent = repo_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        run([
            "gh", "repo", "create", repo_name, "--public",
            "--description", f"{args.lang} Wikipedia corpus for rababa ML training",
            "--clone",
        ], cwd=parent)
        cloned = parent / repo_name.split("/")[-1]
        if cloned.exists() and not repo_dir.exists():
            cloned.rename(repo_dir)

    # Lay out data dir like rababa-tashkeela.
    subdir_prefix = f"{args.lang}wiki"
    subdirs = {"train": f"{subdir_prefix}_train",
               "val": f"{subdir_prefix}_val",
               "test": f"{subdir_prefix}_test"}
    stats: dict[str, int] = {}
    for split, subdir in subdirs.items():
        src = args.data_dir / f"{split}.txt"
        dst_dir = repo_dir / subdir
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / f"{split}.txt"
        shutil.copy2(src, dst)
        with dst.open(encoding="utf-8") as f:
            stats[split] = sum(1 for _ in f)

    # Write README.
    total = sum(stats.values())
    (repo_dir / "README.adoc").write_text(f"""= {args.lang.upper()} Wikipedia corpus for rababa

== Purpose

Plain-text {args.lang} Wikipedia lines used as the MLM pre-training corpus
for rababa's {args.lang} diacritization model.

For Arabic, this augments the gold Tashkeela fine-tune corpus with
~{total:,} lines of unpointed Modern Standard Arabic prose.

For Hebrew, this is the *unpointed* source corpus for distillation —
the rababa Modal distillation pipeline runs each line through the
Dicta Nakdan API to produce pointed labels. The distilled result lives
in a separate `rababa-hebrew-distilled` repo.

== Source

Fetched from the
https://huggingface.co/datasets/wikimedia/wikipedia[wikimedia/wikipedia
dataset on Hugging Face] ({args.lang} config, 20231101 dump) via
`scripts/fetch_wiki_corpus.py` in the main
https://github.com/interscript/rababa[rababa] repo.

== License

Wikipedia text is © Wikipedia contributors, licensed under
https://creativecommons.org/licenses/by-sa/4.0/[CC-BY-SA 4.0].
This compiled corpus follows the same license.

== Stats

Total lines: {total:,}

[cols="1,>1", options="header"]
|===
| Split | Lines
| train | {stats.get('train', 0):,}
| val | {stats.get('val', 0):,}
| test | {stats.get('test', 0):,}
|===

== Layout

----
{subdir_prefix}_train/train.txt
{subdir_prefix}_val/val.txt
{subdir_prefix}_test/test.txt
----
""", encoding="utf-8")

    # Check existing commits.
    has_commits = (
        subprocess.run(["git", "-C", str(repo_dir), "log", "--oneline"],
                       capture_output=True).returncode == 0
    )

    # Stage explicit paths only.
    run(["git", "-C", str(repo_dir), "add",
         "README.adoc",
         f"{subdir_prefix}_train/train.txt",
         f"{subdir_prefix}_val/val.txt",
         f"{subdir_prefix}_test/test.txt"])
    run(["git", "-C", str(repo_dir), "status", "--short"])

    if has_commits:
        branch = "update-corpus"
        run(["git", "-C", str(repo_dir), "checkout", "-b", branch])
        run(["git", "-C", str(repo_dir), "commit",
             "-m", f"Update {args.lang} Wikipedia corpus"])
        run(["git", "-C", str(repo_dir), "push", "-u", "origin", branch])
        run(["gh", "pr", "create", "--repo", repo_name,
             "--title", f"Update {args.lang} Wikipedia corpus",
             "--body", "Automated update of train/val/test splits.",
             "--head", branch])
    else:
        run(["git", "-C", str(repo_dir), "commit",
             "-m", f"Initial {args.lang} Wikipedia corpus ({total:,} lines)"])
        run(["git", "-C", str(repo_dir), "push", "-u", "origin", "main"])

    print(f"\n✓ {repo_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

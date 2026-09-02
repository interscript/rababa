#!/usr/bin/env python3
"""Commit Sefaria corpus to interscript/rababa-sefaria data repo.

Creates the repo if missing, then commits the fetched train/val/test
files in the same layout as interscript/rababa-tashkeela.

Idempotent: if the repo exists and has the data, just updates it.

Usage:
  python scripts/commit_sefaria_corpus.py
  python scripts/commit_sefaria_corpus.py --data-dir data/sefaria-tanakh
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sefaria-tanakh"
REPO_NAME = "interscript/rababa-sefaria"
REPO_DIR = Path(__file__).resolve().parent.parent / ".sefaria-repo"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, print + check."""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def repo_exists() -> bool:
    """Check if the GitHub repo exists."""
    result = subprocess.run(
        ["gh", "repo", "view", REPO_NAME],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def create_repo() -> None:
    """Create the interscript/rababa-sefaria repo on GitHub."""
    if repo_exists():
        print(f"Repo {REPO_NAME} already exists.")
        return
    # gh repo create --clone clones into ./<repo-name> under cwd. We cd first.
    parent = REPO_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)
    run([
        "gh", "repo", "create", REPO_NAME,
        "--public",
        "--description", "Pointed Hebrew corpus from Sefaria used for training rababa Hebrew diacritization",
        "--clone",
    ], cwd=parent)
    # gh clones into parent/rababa-sefaria — rename to our expected path.
    cloned = parent / "rababa-sefaria"
    if cloned.exists() and not REPO_DIR.exists():
        cloned.rename(REPO_DIR)


def clone_repo() -> None:
    """Clone the existing repo to REPO_DIR."""
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["gh", "repo", "clone", REPO_NAME, str(REPO_DIR)])


def write_readme(repo_dir: Path, stats: dict[str, int]) -> None:
    """Write README.adoc following the rababa-tashkeela pattern."""
    readme = repo_dir / "README.adoc"
    total = stats.get("train", 0) + stats.get("val", 0) + stats.get("test", 0)
    content = f"""= Sefaria pointed Hebrew corpus as used by rababa

== Purpose

Pointed (nikud + dagesh + sin) Hebrew text used to train the
https://github.com/interscript/rababa[rababa] Hebrew diacritization
model.

== Source

All text fetched from the
https://www.sefaria.org[Sefaria API] (developers.sefaria.org). Sefaria
is a non-profit organization that makes Jewish texts freely available.

The fetch script lives at
`https://github.com/interscript/rababa/blob/main/scripts/fetch_sefaria_corpus.py[scripts/fetch_sefaria_corpus.py]`.

== Books included

- *Tanakh* (39 books): Torah + Nevi'im + Ketuvim — fully pointed
  Biblical Hebrew.
- *Mishnah* (63 tractates across 6 sedarim): rabbinic legal text.
- *Siddurim* (Ashkenaz, Sefard, Edot HaMizrach): prayer books.

== Splits

This is a *Biblical + Rabbinic Hebrew* corpus, not Modern Hebrew.
Diurnal usage differs from modern (e.g., grammar, vocabulary). For
Modern Hebrew, see the distillation-augmented corpus in
`rababa-modern-hebrew-distilled` (TBD).

== License

Sefaria texts are public domain or various open licenses (CC-BY, CC-0).
See https://www.sefaria.org/texts[Sefaria's licensing page] for
per-text details. This compiled dataset is provided for unencumbered
ML training access.

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
sefaria_train/train.txt
sefaria_val/val.txt
sefaria_test/test.txt
----

Matches the layout of
https://github.com/interscript/rababa-tashkeela[interscript/rababa-tashkeela].
"""
    readme.write_text(content, encoding="utf-8")


def copy_data(data_dir: Path, repo_dir: Path) -> dict[str, int]:
    """Copy train/val/test into the repo's expected subdirs. Returns line counts."""
    subdirs = {
        "train": "sefaria_train",
        "val": "sefaria_val",
        "test": "sefaria_test",
    }
    stats: dict[str, int] = {}
    for split, subdir in subdirs.items():
        src = data_dir / f"{split}.txt"
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        dst_dir = repo_dir / subdir
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / f"{split}.txt"
        shutil.copy2(src, dst)
        with dst.open(encoding="utf-8") as f:
            stats[split] = sum(1 for _ in f)
    return stats


def commit_and_push(repo_dir: Path) -> None:
    """Stage explicit files, commit, push to main (initial commit on a new repo)."""
    # Check if there are existing commits (determines branch strategy).
    has_commits = (
        subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--oneline"],
            capture_output=True,
        ).returncode == 0
    )

    # Stage explicit paths only — never `git add -A`.
    run(["git", "-C", str(repo_dir), "add",
         "README.adoc",
         "sefaria_train/train.txt",
         "sefaria_val/val.txt",
         "sefaria_test/test.txt"])

    # Show staged diff for verification before commit.
    run(["git", "-C", str(repo_dir), "status", "--short"])

    if has_commits:
        # Subsequent commit — branch + PR.
        branch = "update-corpus"
        run(["git", "-C", str(repo_dir), "checkout", "-b", branch])
        run(["git", "-C", str(repo_dir), "commit",
             "-m", "Update Sefaria corpus"])
        run(["git", "-C", str(repo_dir), "push", "-u", "origin", branch])
        run(["gh", "pr", "create",
             "--repo", REPO_NAME,
             "--title", "Update Sefaria corpus",
             "--body", "Automated update of train/val/test splits.",
             "--head", branch])
    else:
        # Initial commit on a new repo.
        run(["git", "-C", str(repo_dir), "commit",
             "-m", "Initial Sefaria corpus (Tanakh + Mishnah + Siddurim)"])
        run(["git", "-C", str(repo_dir), "push", "-u", "origin", "main"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--no-push", action="store_true",
                   help="Prepare commit but don't push")
    args = p.parse_args(argv)

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir {args.data_dir} doesn't exist. Run fetch_sefaria_corpus.py first.",
              file=sys.stderr)
        return 1

    if not (args.data_dir / "train.txt").is_file():
        print(f"ERROR: {args.data_dir}/train.txt missing. Fetch not complete?", file=sys.stderr)
        return 1

    print(f"=== Committing Sefaria corpus from {args.data_dir} ===\n")

    if repo_exists():
        clone_repo()
    else:
        create_repo()

    print(f"\n=== Copying data into {REPO_DIR} ===")
    stats = copy_data(args.data_dir, REPO_DIR)
    print(f"\nStats: {stats}")

    write_readme(REPO_DIR, stats)

    if args.no_push:
        print("\n--no-push: prepared but not committed. Inspect at", REPO_DIR)
        return 0

    print("\n=== Committing + pushing ===")
    commit_and_push(REPO_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

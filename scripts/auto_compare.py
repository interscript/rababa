#!/usr/bin/env python3
"""Pull all rababa training metrics from Modal and run N-way comparison.

Usage:
    python scripts/auto_compare.py hebrew    # compare all Hebrew variants
    python scripts/auto_compare.py arabic    # compare all Arabic variants
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


HEBREW_RUNS = [
    ("baseline v0.6.0", "rababa_hebrew"),
    ("DS-V4 Tier 1", "rababa_hebrew_dsv4"),
    ("ResFormer", "rababa_hebrew_resformer"),
    ("ResFormer reg", "rababa_hebrew_resformer_reg"),
    ("AdaMuon+Nor", "rababa_hebrew_adamuon"),
]

ARABIC_RUNS = [
    ("baseline v0.6.0", "rababa_arabic_pro"),
    ("DS-V4 Tier 1", "rababa_arabic_pro_dsv4"),
    ("ResFormer", "rababa_arabic_pro_resformer"),
    ("AdaMuon+Nor", "rababa_arabic_pro_adamuon"),
]


def pull_metrics(task: str, dest: Path) -> bool:
    """Pull metrics-{task}-train.jsonl from rababa-checkpoints volume.

    Some old runs use `metrics-{task}.jsonl` (no -train suffix) — try both.
    """
    candidates = [
        f"metrics/metrics-{task}-train.jsonl",
        f"metrics/metrics-{task}.jsonl",
    ]
    for remote in candidates:
        result = subprocess.run(
            ["modal", "volume", "get", "rababa-checkpoints", remote, str(dest)],
            capture_output=True, text=True,
        )
        if dest.is_file() and dest.stat().st_size > 0:
            return True
    return dest.is_file() and dest.stat().st_size > 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=["hebrew", "arabic"], help="Which family to compare")
    args = p.parse_args()

    runs = HEBREW_RUNS if args.family == "hebrew" else ARABIC_RUNS

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd_args = []
        seen_labels = {}
        for label, task in runs:
            dest = Path(tmpdir) / f"{task}.jsonl"
            ok = pull_metrics(task, dest)
            if not ok:
                print(f"  WARNING: no metrics for {task}", file=sys.stderr)
                continue
            # Disambiguate duplicate labels (unlikely but safe).
            base = label
            n = seen_labels.get(base, 0) + 1
            seen_labels[base] = n
            label_unique = base if n == 1 else f"{base}#{n}"
            cmd_args.extend(["--label", label_unique, "--metrics", str(dest)])

        if not cmd_args:
            print("Error: no metrics files found", file=sys.stderr)
            return 2

        # Delegate to compare_techniques.py.
        result = subprocess.run(
            ["python", "scripts/compare_techniques.py", *cmd_args],
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())

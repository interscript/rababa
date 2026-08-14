#!/usr/bin/env python3
"""N-way comparison: baseline vs DS-V4 vs ResFormer vs ablations.

Reads multiple metrics JSONL files and prints a side-by-side comparison
table of train_loss, val_loss per epoch, plus summary stats.

Usage:
    python scripts/compare_techniques.py \\
        --label "Hebrew baseline" --metrics /tmp/metrics-rababa_hebrew-train.jsonl \\
        --label "Hebrew DS-V4"     --metrics /tmp/metrics-rababa_hebrew_dsv4-train.jsonl \\
        --label "Hebrew ResFormer" --metrics /tmp/metrics-rababa_hebrew_resformer-train.jsonl \\
        --label "Hebrew ResOnly"   --metrics /tmp/metrics-rababa_hebrew_resformer_only-train.jsonl \\
        --label "Hebrew AdaMuon"   --metrics /tmp/metrics-rababa_hebrew_adamuon-train.jsonl

If a metrics file has multiple runs (e.g., from NaN-recovery restart),
uses the LAST contiguous run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    """Load JSONL metrics file. Returns list of epoch dicts."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def last_contiguous_run(metrics: list[dict]) -> list[dict]:
    """If metrics has multiple epoch-0 entries (from NaN-recovery restarts),
    return only the last contiguous run."""
    if not metrics:
        return []
    last_run_start = 0
    for i, m in enumerate(metrics):
        if m.get("epoch") == 0 and i > 0:
            last_run_start = i
    return metrics[last_run_start:]


def fmt(x: float | None, width: int = 8, prec: int = 4) -> str:
    if x is None:
        return " " * width
    if isinstance(x, float) and (x != x or abs(x) > 1e6):
        return f"{'NaN':>{width}}"
    return f"{x:>{width}.{prec}f}"


def best_val(metrics: list[dict]) -> float:
    """Lowest non-NaN val_loss."""
    vals = [m["val_loss"] for m in metrics
            if m.get("val_loss") is not None
            and m["val_loss"] == m["val_loss"]
            and abs(m["val_loss"]) < 1e6]
    return min(vals) if vals else float("inf")


def final_val(metrics: list[dict]) -> float | None:
    """Last non-NaN val_loss."""
    for m in reversed(metrics):
        v = m.get("val_loss")
        if v is not None and v == v and abs(v) < 1e6:
            return v
    return None


def stability(metrics: list[dict]) -> tuple[int, float]:
    """(NaN-count, stddev) of val_loss."""
    vals = [m["val_loss"] for m in metrics
            if m.get("val_loss") is not None
            and m["val_loss"] == m["val_loss"]
            and abs(m["val_loss"]) < 1e6]
    n_nans = sum(1 for m in metrics
                 if m.get("val_loss") is not None
                 and (m["val_loss"] != m["val_loss"]
                      or abs(m["val_loss"]) > 1e6))
    if len(vals) < 2:
        return n_nans, 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return n_nans, var ** 0.5


def compare(runs: list[tuple[str, list[dict]]]) -> None:
    """Print N-way comparison table. `runs` is [(label, metrics), ...]."""
    n_runs = len(runs)
    print(f"\n{'=' * (40 + n_runs * 14)}")
    print(f"N-way Comparison ({n_runs} runs)")
    print(f"{'=' * (40 + n_runs * 14)}\n")

    if any(not m for _, m in runs):
        for label, m in runs:
            if not m:
                print(f"  WARNING: no metrics for '{label}'")

    # Header: epoch | label1 train | label1 val | label2 train | label2 val | ...
    header_parts = [f"{'epoch':>5}"]
    for label, _ in runs:
        # Truncate label to 12 chars for column width.
        short = label[:12]
        header_parts.append(f"{short + ' train':>13}")
        header_parts.append(f"{short + ' val':>10}")
    print(" | ".join(header_parts))
    print("-" * (len(" | ".join(header_parts))))

    n_max = max((len(m) for _, m in runs), default=0)
    for i in range(n_max):
        epoch = None
        for _, m in runs:
            if i < len(m):
                epoch = m[i].get("epoch")
                if epoch is not None:
                    break
        row_parts = [f"{epoch:>5}"]
        for _, m in runs:
            t = m[i].get("train_loss") if i < len(m) else None
            v = m[i].get("val_loss") if i < len(m) else None
            row_parts.append(fmt(t, 13))
            row_parts.append(fmt(v, 10))
        print(" | ".join(row_parts))

    # Summary
    print(f"\n{'Summary':>40}")
    print("-" * 60)
    best_vals = [(label, best_val(m)) for label, m in runs]
    final_vals = [(label, final_val(m)) for label, m in runs]
    stabilities = [(label, *stability(m)) for label, m in runs]

    best_overall = min(best_vals, key=lambda x: x[1]) if best_vals else None

    print(f"\n  {'label':<30} {'best val':>10} {'final val':>10} {'stddev':>8} {'NaNs':>5}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 5}")
    for (label, bv), (_, fv), (label2, nans, std) in zip(best_vals, final_vals, stabilities):
        bv_str = f"{bv:.4f}" if bv != float("inf") else "  N/A"
        fv_str = f"{fv:.4f}" if fv is not None else "  N/A"
        print(f"  {label:<30} {bv_str:>10} {fv_str:>10} {std:>8.4f} {nans:>5}")

    # Verdict
    if best_overall is not None and best_overall[1] != float("inf"):
        winner_label, winner_val = best_overall
        print(f"\n  Winner: {winner_label} (best val_loss = {winner_val:.4f})")

        # Deltas vs first run (assumed baseline).
        if runs and runs[0][1]:
            baseline_label, baseline_val = best_vals[0]
            if baseline_val != float("inf"):
                for label, bv in best_vals[1:]:
                    if bv != float("inf"):
                        diff = bv - baseline_val
                        pct = (diff / max(baseline_val, 1e-6)) * 100
                        sign = "+" if diff >= 0 else ""
                        verdict = "BETTER" if diff < 0 else "WORSE" if diff > 0 else "TIE"
                        print(f"  {label:<30} Δ={sign}{diff:.4f} ({sign}{pct:.2f}%) [{verdict}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Accept --label/--metrics pairs in order.
    p.add_argument("--label", action="append", default=[],
                   help="Label for the next --metrics (one per metrics file, in order)")
    p.add_argument("--metrics", action="append", default=[],
                   help="Path to a metrics JSONL file (one per label, in order)")
    args = p.parse_args()

    if len(args.label) != len(args.metrics):
        print(f"Error: {len(args.label)} labels vs {len(args.metrics)} metrics files",
              file=sys.stderr)
        print("Provide one --label for each --metrics, in order.", file=sys.stderr)
        return 2
    if not args.label:
        print("Error: provide at least one --label/--metrics pair", file=sys.stderr)
        return 2

    runs = []
    for label, path_str in zip(args.label, args.metrics):
        m = last_contiguous_run(load_metrics(Path(path_str)))
        runs.append((label, m))

    compare(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

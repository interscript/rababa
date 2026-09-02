#!/usr/bin/env python3
"""A/B comparison: baseline v0.6.0 vs DS-V4-Flash Tier 1 techniques.

Reads two metrics JSONL files and prints a side-by-side comparison table
of train_loss, val_loss per epoch, plus summary stats.

Usage:
    python scripts/compare_dsv4_ab.py \
        --baseline /tmp/metrics-rababa_hebrew-train.jsonl \
        --dsv4 /tmp/metrics-rababa_hebrew_dsv4-train.jsonl \
        --label-baseline "Hebrew v0.6.0 (baseline)" \
        --label-dsv4 "Hebrew v0.6.1 (DS-V4-Flash)"

If the baseline file has multiple runs (e.g., from a NaN-recovery restart),
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
    if isinstance(x, float) and (x != x or abs(x) > 1e6):  # NaN or huge
        return f"{'NaN':>{width}}"
    return f"{x:>{width}.{prec}f}"


def compare(
    baseline: list[dict],
    dsv4: list[dict],
    label_baseline: str,
    label_dsv4: str,
) -> None:
    print(f"\n{'='*72}")
    print(f"A/B Comparison: {label_baseline} vs {label_dsv4}")
    print(f"{'='*72}\n")

    if not baseline and not dsv4:
        print("No metrics found in either file.")
        return

    # Header
    print(f"{'epoch':>5} | {'baseline train':>14} | {'baseline val':>12} | "
          f"{'dsv4 train':>11} | {'dsv4 val':>9} | {'val Δ':>9}")
    print("-" * 72)

    n_base = len(baseline)
    n_dsv4 = len(dsv4)
    n_max = max(n_base, n_dsv4)

    base_best_val = float("inf")
    dsv4_best_val = float("inf")
    base_final_val = None
    dsv4_final_val = None

    for i in range(n_max):
        epoch_b = baseline[i]["epoch"] if i < n_base else None
        epoch_d = dsv4[i]["epoch"] if i < n_dsv4 else None
        # Epoch numbers may not align if DS-V4 is still running. Use the
        # epoch field from whichever side has data.
        epoch = epoch_b if epoch_b is not None else epoch_d

        b_train = baseline[i].get("train_loss") if i < n_base else None
        b_val = baseline[i].get("val_loss") if i < n_base else None
        d_train = dsv4[i].get("train_loss") if i < n_dsv4 else None
        d_val = dsv4[i].get("val_loss") if i < n_dsv4 else None

        if b_val is not None and b_val == b_val:
            base_best_val = min(base_best_val, b_val)
            base_final_val = b_val
        if d_val is not None and d_val == d_val:
            dsv4_best_val = min(dsv4_best_val, d_val)
            dsv4_final_val = d_val

        delta = ""
        if b_val is not None and d_val is not None and b_val == b_val and d_val == d_val:
            diff = d_val - b_val
            sign = "+" if diff >= 0 else ""
            delta = f"{sign}{diff:>+8.4f}"

        print(f"{epoch:>5} | {fmt(b_train, 14)} | {fmt(b_val, 12)} | "
              f"{fmt(d_train, 11)} | {fmt(d_val, 9)} | {delta:>9}")

    # Summary
    print(f"\n{'Summary':>72}")
    print("-" * 72)
    if base_best_val != float("inf"):
        print(f"  baseline best val_loss: {base_best_val:.4f}")
    if dsv4_best_val != float("inf"):
        print(f"  dsv4     best val_loss: {dsv4_best_val:.4f}")
    if base_best_val != float("inf") and dsv4_best_val != float("inf"):
        diff = dsv4_best_val - base_best_val
        pct = (diff / max(base_best_val, 1e-6)) * 100
        sign = "+" if diff >= 0 else ""
        verdict = "BETTER" if diff < 0 else "WORSE" if diff > 0 else "TIE"
        print(f"  delta (dsv4 - base):    {sign}{diff:.4f}  ({sign}{pct:.2f}%)  [{verdict}]")
    if base_final_val is not None:
        print(f"  baseline final val_loss: {base_final_val:.4f}")
    if dsv4_final_val is not None:
        print(f"  dsv4     final val_loss: {dsv4_final_val:.4f}")

    # Stability: count NaN val_loss and variance
    def _stability(metrics: list[dict]) -> tuple[int, float]:
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

    base_nans, base_std = _stability(baseline)
    dsv4_nans, dsv4_std = _stability(dsv4)
    print(f"\n  stability (val_loss stddev):")
    print(f"    baseline: {base_std:.4f}  ({base_nans} NaN/spike epochs)")
    print(f"    dsv4:     {dsv4_std:.4f}  ({dsv4_nans} NaN/spike epochs)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--dsv4", required=True, type=Path)
    p.add_argument("--label-baseline", default="baseline")
    p.add_argument("--label-dsv4", default="dsv4")
    args = p.parse_args()

    base = last_contiguous_run(load_metrics(args.baseline))
    dsv4 = last_contiguous_run(load_metrics(args.dsv4))
    if not base:
        print(f"No baseline metrics at {args.baseline}", file=sys.stderr)
    if not dsv4:
        print(f"No DS-V4 metrics at {args.dsv4}", file=sys.stderr)
    compare(base, dsv4, args.label_baseline, args.label_dsv4)
    return 0


if __name__ == "__main__":
    sys.exit(main())

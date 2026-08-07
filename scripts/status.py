#!/usr/bin/env python3
"""Query Modal volumes for sprint progress. Safe to run from anywhere.

Use this to reconnect after a disconnect:
    python scripts/status.py

Shows:
  - Stage status index (which stages marked done)
  - Latest checkpoint per task (epoch / best_val_loss)
  - Volume log files (newest first)

This script NEVER modifies the volume — it only reads. Pair with
`python scripts/train_all.py` to skip completed stages on re-run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "rababa"


def modal_volume_ls(volume: str, path: str = "/") -> list[str]:
    """Return stdout lines of `modal volume ls <volume> <path>`."""
    try:
        result = subprocess.run(
            ["modal", "volume", "ls", volume, path],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if result.returncode != 0:
            return []
        return result.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def modal_volume_get(volume: str, remote_path: str, local_path: str) -> bool:
    """`modal volume get`. Returns True on success."""
    result = subprocess.run(
        ["modal", "volume", "get", volume, remote_path, local_path],
        capture_output=True, text=True, check=False, timeout=60,
    )
    return result.returncode == 0


def fetch_status_json() -> dict:
    """Fetch /checkpoints/_status.json to a temp file and parse it."""
    tmp = Path("/tmp/rababa-status.json")
    if tmp.exists():
        tmp.unlink()
    modal_volume_get(f"{APP_NAME}-checkpoints", "/checkpoints/_status.json", str(tmp))
    if not tmp.is_file():
        return {}
    try:
        return json.loads(tmp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fetch_dir_listing(volume: str, path: str) -> list[str]:
    """Return listing of a volume path. Returns [] on error."""
    return modal_volume_ls(volume, path)


def format_status(status: dict) -> str:
    if not status:
        return "(no stage status index yet — no stages have completed)"
    stages = status.get("stages", {})
    if not stages:
        return "(no stages recorded)"
    out = []
    for name in sorted(stages.keys()):
        entry = stages[name]
        done = "✓" if entry.get("done") else "✗"
        ts = entry.get("ts", 0)
        from datetime import datetime
        when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        err = f"  ERROR: {entry['error'][:80]}" if entry.get("error") else ""
        out.append(f"  [{done}] {name:30s}  {when}{err}")
    return "\n".join(out)


def format_checkpoints(volume: str, task: str) -> str:
    listing = fetch_dir_listing(volume, f"/checkpoints/{task}/run-001")
    if not listing:
        return f"  (no checkpoints at /checkpoints/{task}/run-001)"
    ckpts = [l for l in listing if "checkpoint-epoch" in l or "best.pt" in l]
    if not ckpts:
        return f"  (no checkpoints yet at /checkpoints/{task}/run-001)"
    head = sorted(ckpts)[:8]
    tail = sorted(ckpts)[-3:] if len(ckpts) > 8 else []
    out = ["  " + l for l in head]
    if tail:
        out.append("  ...")
        out.extend("  " + l for l in tail)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default=None,
                   help="Show checkpoints for a specific task (e.g., rababa_arabic_pro)")
    args = p.parse_args(argv)

    print(f"=== {APP_NAME} sprint status ===\n")

    print("--- Stage status index (/checkpoints/_status.json) ---")
    print(format_status(fetch_status_json()))
    print()

    print("--- Checkpoints per task ---")
    tasks = [args.task] if args.task else [
        "rababa_arabic_pro_pretrain",
        "rababa_arabic_pro",
        "rababa_arabic_pretrain",
        "rababa_arabic",
        "rababa_hebrew_pretrain",
        "rababa_hebrew",
    ]
    for task in tasks:
        print(f"  {task}:")
        print(format_checkpoints(f"{APP_NAME}-checkpoints", task))
        print()

    print("--- Models volume (/models) ---")
    listing = fetch_dir_listing(f"{APP_NAME}-models", "/models")
    for line in listing[:20]:
        print(f"  {line}")
    if len(listing) > 20:
        print(f"  ... ({len(listing) - 20} more)")
    print()

    print("--- Tips ---")
    print("  To skip completed stages:   python scripts/train_all.py")
    print("  To pull a checkpoint:       modal volume get rababa-checkpoints \\")
    print("                                 /checkpoints/<task>/run-001/best.pt ./")
    print("  To pull the latest logs:    modal volume get rababa-checkpoints \\")
    print("                                 /logs/<task>.log ./")
    return 0


if __name__ == "__main__":
    sys.exit(main())

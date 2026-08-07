#!/usr/bin/env python3
"""One-shot SOTA sprint progress report.

Pulls every available signal from Modal into a single readable report:

  1. App state         — modal app list (what's running right now)
  2. Stage status      — /checkpoints/_status.json (what's done / failed)
  3. Pipeline log      — /checkpoints/logs/sota_pipeline-*.log (timestamps)
  4. Checkpoints       — /checkpoints/<task>/run-001/checkpoint-epoch-N.pt
                         (latest epoch written per task)
  5. Models            — /models/<task>/*.onnx / *.tflite (final artifacts)

Caches pulled files under /tmp/rababa-status/ so re-runs are fast.

Usage:
    python scripts/status.py                  # all signals
    python scripts/status.py --task rababa_arabic_pro
    python scripts/status.py --watch          # refresh every 60s
    python scripts/status.py --json           # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "rababa"
CACHE_DIR = Path(tempfile.gettempdir()) / "rababa-status"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---- Modal wrappers ---------------------------------------------------


def modal_volume_ls(volume: str, path: str = "/") -> list[str]:
    """Return non-empty stdout lines of `modal volume ls <volume> <path>`."""
    try:
        r = subprocess.run(
            ["modal", "volume", "ls", volume, path],
            capture_output=True, text=True, check=False, timeout=20,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if r.returncode != 0:
        return []
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def modal_volume_get(volume: str, remote: str, local: Path) -> bool:
    """`modal volume get` to a local path. Returns True on success."""
    if local.exists():
        local.unlink()
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["modal", "volume", "get", volume, remote, str(local.parent) + "/"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    return r.returncode == 0 and local.is_file()


def modal_app_list_raw() -> str:
    try:
        r = subprocess.run(
            ["modal", "app", "list"], capture_output=True, text=True,
            check=False, timeout=20,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    return r.stdout if r.returncode == 0 else ""


# ---- Pullers ----------------------------------------------------------


def pull_status_json() -> dict:
    """Pull /checkpoints/_status.json → parsed dict."""
    local = CACHE_DIR / "_status.json"
    if not modal_volume_get(f"{APP_NAME}-checkpoints", "/_status.json", local):
        return {}
    try:
        return json.loads(local.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def pull_pipeline_log(task: str, tail_n: int = 30) -> str:
    """Pull pipeline log → last N lines as string."""
    remote = f"/logs/sota_pipeline-{task}.log"
    local = CACHE_DIR / f"sota_pipeline-{task}.log"
    if not modal_volume_get(f"{APP_NAME}-checkpoints", remote, local):
        return ""
    try:
        lines = local.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-tail_n:])
    except OSError:
        return ""


def list_checkpoints(task: str) -> list[str]:
    """List checkpoint files for a task's run-001 directory."""
    listing = modal_volume_ls(
        f"{APP_NAME}-checkpoints", f"/{task}/run-001"
    )
    # Filter to checkpoint-epoch-N.pt and best.pt; sort by epoch
    epochs: list[tuple[int, str]] = []
    best = None
    for ln in listing:
        parts = ln.split()
        if not parts:
            continue
        name = parts[-1].split("/")[-1]
        if name.startswith("checkpoint-epoch-") and name.endswith(".pt"):
            try:
                n = int(name.removeprefix("checkpoint-epoch-").removesuffix(".pt"))
                epochs.append((n, name))
            except ValueError:
                pass
        elif name == "best.pt":
            best = name
    epochs.sort()
    out = [name for _, name in epochs]
    if best:
        out.append("★ " + best)
    return out


def list_models(task: str) -> list[str]:
    """List exported artifacts for a task under /models."""
    listing = modal_volume_ls(f"{APP_NAME}-models", f"/{task}")
    out: list[str] = []
    for ln in listing:
        parts = ln.split()
        if not parts:
            continue
        out.append(parts[-1].split("/")[-1])
    return out


def parse_app_states(stdout: str) -> list[dict[str, str]]:
    """Parse `modal app list` table output into list of dicts.

    Columns (Modal CLI format): App ID | Description | State | Tasks | Created
    """
    rows: list[dict[str, str]] = []
    seen_header = False
    for ln in stdout.splitlines():
        if "┃" not in ln and "│" not in ln:
            continue
        # Split on the vertical bar character regardless of which one.
        sep = "┃" if "┃" in ln else "│"
        cells = [c.strip() for c in ln.split(sep)]
        # First/last cells are typically empty (from leading/trailing bars).
        cells = [c for c in cells if c != ""]
        if not seen_header:
            seen_header = True
            continue  # skip header row
        if len(cells) >= 5 and not cells[0].startswith("─") and not cells[0].startswith("═"):
            rows.append({
                "app_id": cells[0],
                "description": cells[1],
                "state": cells[2],
                "tasks": cells[3],
                "created": cells[4],
            })
    return rows


# ---- Report -----------------------------------------------------------


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _stage_label(stage: dict) -> tuple[str, str]:
    if stage.get("done"):
        return "✓ DONE", "\033[32m"
    if stage.get("error"):
        return "✗ FAIL", "\033[31m"
    return "⏳ RUN", "\033[33m"


def report(task: str, tail_n: int = 30) -> dict[str, object]:
    """Print full report. Returns dict for --json mode."""
    # 1. apps
    apps_raw = modal_app_list_raw()
    apps = [a for a in parse_app_states(apps_raw) if a.get("description") == APP_NAME]
    live = [a for a in apps if "ephemeral" in a.get("state", "") or "running" in a.get("state", "")]

    # 2. status JSON
    status = pull_status_json()
    stages = status.get("stages", {})

    # 3. log tail
    log_tail = pull_pipeline_log(task, tail_n=tail_n)

    # 4. checkpoints
    pretrain_ckpts = list_checkpoints(f"{task}_pretrain")
    train_ckpts = list_checkpoints(task)

    # 5. models
    artifacts = list_models(task)

    out = {
        "task": task,
        "apps": apps,
        "stages": stages,
        "log_tail": log_tail,
        "pretrain_checkpoints": pretrain_ckpts,
        "train_checkpoints": train_ckpts,
        "artifacts": artifacts,
    }

    # Print human report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== rababa SOTA sprint status  @ {now} ===")
    print(f"task: {task}\n")

    print("--- Apps ---")
    if not apps:
        print("  (none — nothing is running)")
    for a in apps[:5]:
        print(f"  {a['app_id']}  state={a['state']:<25} tasks={a['tasks']}  {a['created']}")
    if live:
        print(f"  → LIVE: https://modal.com/apps/ronaldtse/main/{live[0]['app_id']}")
    print()

    print("--- Stages ---")
    if not stages:
        print("  (no stage index yet — orchestrator hasn't started or no stages done)")
    for name in sorted(stages.keys()):
        s = stages[name]
        label, color = _stage_label(s)
        reset = "\033[0m" if color else ""
        ts = _fmt_ts(s.get("ts"))
        err = f"  err: {s['error'][:100]}" if s.get("error") else ""
        lines = ""
        if s.get("files"):
            parts = []
            for split, info in s["files"].items():
                parts.append(f"{split}={info.get('lines', '?'):,}")
            lines = "  (" + " ".join(parts) + ")"
        print(f"  {color}[{label}]{reset} {name:<10} {ts}{lines}{err}")
    print()

    print(f"--- Pretrain checkpoints  ({task}_pretrain/run-001) ---")
    if pretrain_ckpts:
        for c in pretrain_ckpts[-6:]:
            print(f"  {c}")
    else:
        print("  (none yet)")
    print()

    print(f"--- Supervised checkpoints  ({task}/run-001) ---")
    if train_ckpts:
        for c in train_ckpts[-6:]:
            print(f"  {c}")
    else:
        print("  (none yet)")
    print()

    print(f"--- Exported artifacts  (/models/{task}) ---")
    if artifacts:
        for a in artifacts:
            print(f"  {a}")
    else:
        print("  (none yet)")
    print()

    print("--- Pipeline log tail ---")
    if log_tail:
        for ln in log_tail.splitlines()[-tail_n:]:
            print(f"  {ln}")
    else:
        print("  (log file not yet on volume)")
    print()

    print("--- Reconnect commands ---")
    print(f"  python scripts/status.py                       # this report")
    print(f"  python scripts/status.py --watch               # refresh every 60s")
    print(f"  modal volume get {APP_NAME}-models /{task}/ ./models/  # pull artifacts when ready")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default="rababa_arabic_pro")
    p.add_argument("--watch", action="store_true",
                   help="Refresh every 60s until Ctrl-C")
    p.add_argument("--interval", type=int, default=60,
                   help="Refresh interval (seconds, default 60)")
    p.add_argument("--tail", type=int, default=30,
                   help="Number of log lines to show (default 30)")
    p.add_argument("--json", action="store_true",
                   help="Print machine-readable JSON instead of human report")
    args = p.parse_args(argv)

    if args.json:
        result = report(args.task, tail_n=0)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.watch:
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                report(args.task, tail_n=args.tail)
                print(f"\n  (refreshing in {args.interval}s — Ctrl-C to quit)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  stopped.")
        return 0

    report(args.task, tail_n=args.tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())

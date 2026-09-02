#!/usr/bin/env python3
"""End-to-end rababa training orchestration.

Runs the full pipeline for both Arabic and Hebrew:
  1. fetch_data        — verify corpus is on the Modal volume
  2. pretrain          — MLM char-level pretraining (~6h A100 / lang)
  3. train             — Tier 1 supervised fine-tune (~3h A100 / lang)
  4. export_onnx       — fp32 + int8 ONNX artifacts
  5. export_tflite     — fp32 TFLite artifact (for LiteRT.js)
  6. pull              — download artifacts from Modal to ./models/
  7. benchmark         — DER vs legacy 2021 baseline

Each run is logged under runs/<timestamp>/:
  run.log               — full streamed output
  stage-<n>-<name>.log  — per-stage log
  summary.json          — structured status of all stages
  benchmark-*.json      — benchmark result files

Usage:
  python scripts/train_all.py                    # run everything (~18h A100)
  python scripts/train_all.py --only-lang arabic # skip Hebrew
  python scripts/train_all.py --skip-to 4        # resume from stage 4
  python scripts/train_all.py --dry-run          # print commands, don't execute

Prerequisites:
  - `modal token new` run once to authenticate
  - Paid Modal account (this run costs ~$37 in compute)
  - For final benchmark: legacy ONNX models in models-data/ (already in repo)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
MODAL_APP = "modal_app.py"


@dataclass
class Stage:
    """A single pipeline stage. `index` is 1-based for human-friendly logging."""
    index: int
    name: str
    description: str
    estimated_minutes: int
    command: list[str]
    optional: bool = False
    artifacts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "description": self.description,
            "estimated_minutes": self.estimated_minutes,
            "command": self.command,
            "optional": self.optional,
        }


def build_stages(skip_hebrew: bool, no_pull: bool, no_benchmark: bool, skip_arabic_pro: bool = False) -> list[Stage]:
    """Construct the ordered list of pipeline stages."""
    python = sys.executable
    env = {**os.environ, "PYTHONPATH": "src"}

    stages: list[Stage] = []
    idx = 1

    # ---- Arabic ----
    stages.append(Stage(
        index=idx, name="fetch_arabic",
        description="Verify Tashkeela corpus on Modal volume",
        estimated_minutes=2,
        command=["modal", "run", MODAL_APP + "::fetch_data", "--task", "rababa_arabic"],
    )); idx += 1

    stages.append(Stage(
        index=idx, name="pretrain_arabic",
        description="MLM char-level pretrain Arabic (~6h A100, ~$12)",
        estimated_minutes=360,
        command=["modal", "run", MODAL_APP + "::pretrain", "--task", "rababa_arabic_pretrain"],
    )); idx += 1

    stages.append(Stage(
        index=idx, name="train_arabic",
        description="Tier 1 supervised fine-tune Arabic (~3h A100, ~$6)",
        estimated_minutes=180,
        command=[
            "modal", "run", MODAL_APP + "::train",
            "--task", "rababa_arabic",
            "--init-from-pretrain", "/checkpoints/rababa_arabic_pretrain/run-001/best.pt",
        ],
    )); idx += 1

    stages.append(Stage(
        index=idx, name="export_arabic",
        description="Export Arabic to ONNX + TFLite (~30m A10G, ~$0.50)",
        estimated_minutes=30,
        command=[
            "modal", "run", MODAL_APP + "::export_onnx",
            "--task", "rababa_arabic", "--version", "v0.1.0",
        ],
        # TFLite export appended at run time (two sub-commands).
        artifacts=[
            "/models/rababa_arabic/rababa_arabic-v0.1.0-fp32.onnx",
            "/models/rababa_arabic/rababa_arabic-v0.1.0-q8.onnx",
        ],
    )); idx += 1

    # ---- Arabic Pro (12 layers, 768 dim, ~50M params) ----
    # Optional, larger encoder aimed at SOTA-level DER. Trained on the
    # Sadeed-cleaned FULL Tashkeela corpus (497K chunks, ~27M words).
    if not skip_arabic_pro:
        stages.append(Stage(
            index=idx, name="pretrain_arabic_pro",
            description="MLM char-level pretrain Arabic Pro (~12h A100, ~$24)",
            estimated_minutes=720,
            command=["modal", "run", MODAL_APP + "::pretrain",
                     "--task", "rababa_arabic_pro_pretrain"],
        )); idx += 1

        stages.append(Stage(
            index=idx, name="train_arabic_pro",
            description="Tier 1 supervised fine-tune Arabic Pro (~6h A100, ~$12)",
            estimated_minutes=360,
            command=[
                "modal", "run", MODAL_APP + "::train",
                "--task", "rababa_arabic_pro",
                "--init-from-pretrain", "/checkpoints/rababa_arabic_pro_pretrain/run-001/best.pt",
            ],
        )); idx += 1

        stages.append(Stage(
            index=idx, name="export_arabic_pro",
            description="Export Arabic Pro to ONNX + TFLite",
            estimated_minutes=30,
            command=[
                "modal", "run", MODAL_APP + "::export_onnx",
                "--task", "rababa_arabic_pro", "--version", "v0.1.0",
            ],
            artifacts=[
                "/models/rababa_arabic_pro/rababa_arabic_pro-v0.1.0-fp32.onnx",
                "/models/rababa_arabic_pro/rababa_arabic_pro-v0.1.0-q8.onnx",
            ],
        )); idx += 1

    # ---- Hebrew ----
    if not skip_hebrew:
        stages.append(Stage(
            index=idx, name="fetch_hebrew",
            description="Fetch Nakdimon corpus from GitHub (first run only)",
            estimated_minutes=5,
            command=["modal", "run", MODAL_APP + "::fetch_data", "--task", "rababa_hebrew"],
        )); idx += 1

        stages.append(Stage(
            index=idx, name="pretrain_hebrew",
            description="MLM char-level pretrain Hebrew (~6h A100, ~$12)",
            estimated_minutes=360,
            command=["modal", "run", MODAL_APP + "::pretrain", "--task", "rababa_hebrew_pretrain"],
        )); idx += 1

        stages.append(Stage(
            index=idx, name="train_hebrew",
            description="Tier 1 supervised fine-tune Hebrew (~3h A100, ~$6)",
            estimated_minutes=180,
            command=[
                "modal", "run", MODAL_APP + "::train",
                "--task", "rababa_hebrew",
                "--init-from-pretrain", "/checkpoints/rababa_hebrew_pretrain/run-001/best.pt",
            ],
        )); idx += 1

        stages.append(Stage(
            index=idx, name="export_hebrew",
            description="Export Hebrew to ONNX + TFLite",
            estimated_minutes=30,
            command=[
                "modal", "run", MODAL_APP + "::export_onnx",
                "--task", "rababa_hebrew", "--version", "v0.1.0",
            ],
        )); idx += 1

    # ---- Pull + benchmark ----
    if not no_pull:
        stages.append(Stage(
            index=idx, name="pull",
            description="Pull model artifacts from Modal to ./models/",
            estimated_minutes=5,
            command=["modal", "volume", "ls", "rababa-models", "/models"],
            optional=True,
        )); idx += 1

    if not no_benchmark:
        stages.append(Stage(
            index=idx, name="benchmark_arabic",
            description="Benchmark new Arabic v0.1.0 vs legacy 2021 (DER gate ≤ 4.52%)",
            estimated_minutes=2,
            command=[
                python, "-m", "rababa.benchmark",
                "--onnx", "models/rababa_arabic/rababa_arabic-v0.1.0-q8.onnx",
                "--output", "benchmark-v0.1.0-arabic.json",
            ],
            optional=True,
        )); idx += 1

    return stages


def log(msg: str, *, end: str = "\n", flush: bool = True) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", end=end, file=sys.stdout, flush=flush)


def execute_stage(stage: Stage, log_path: Path, dry_run: bool) -> tuple[bool, float]:
    """Execute one stage. Returns (success, elapsed_seconds).

    IDEMPOTENCY: checks `rababa-checkpoints` volume's `/checkpoints/_status.json`
    for a `done` marker on this stage's name. If present and not --force, the
    stage is skipped. After a successful run, the stage is marked done on the
    volume so future invocations skip it (resilient to disconnect).
    """
    # Idempotency: skip if already done on the volume.
    if not dry_run and os.environ.get("RABABA_FORCE") != "1":
        try:
            from src.rababa.training.resume import is_stage_done
            vol_root = Path("/checkpoints")
            if vol_root.is_dir() and is_stage_done(vol_root, stage.name):
                log(f"  ⏭ skipped (already marked done on volume)")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n=== {stage.index} {stage.name} — SKIPPED (done) ===\n")
                return True, 0.0
        except ImportError:
            pass  # resume module not on path (local dev) — fall through

    commands = [stage.command]
    # Special-case: export stages run TWO sub-commands (ONNX + TFLite).
    if stage.name.startswith("export_"):
        lang = stage.name.split("_", 1)[1]
        task = f"rababa_{lang}"
        tflite_cmd = [
            "modal", "run", MODAL_APP + "::export_tflite",
            "--task", task, "--version", "v0.1.0",
        ]
        commands.append(tflite_cmd)
    # Special-case: pull stage pulls both languages.
    elif stage.name == "pull":
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        commands = []
        for lang in ("arabic", "hebrew"):
            commands.append([
                "modal", "volume", "get",
                "rababa-models", f"/models/rababa_{lang}/", str(models_dir) + "/",
            ])

    start = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== {stage.index} {stage.name} — {datetime.now().isoformat()} ===\n")
        f.flush()

    for cmd_idx, cmd in enumerate(commands, start=1):
        cmd_str = "$ " + " ".join(cmd)
        log(f"  [{cmd_idx}/{len(commands)}] {cmd_str}")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{cmd_str}\n")
            f.flush()

        if dry_run:
            continue

        try:
            env = {**os.environ, "PYTHONPATH": "src"}
            proc = subprocess.run(
                cmd, cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, check=False,
            )
        except FileNotFoundError as e:
            log(f"  ✗ command not found: {e}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"COMMAND NOT FOUND: {e}\n")
            return False, time.time() - start

        # Stream + capture
        with log_path.open("a", encoding="utf-8") as f:
            f.write(proc.stdout)
            f.flush()

        if proc.returncode != 0:
            elapsed = time.time() - start
            log(f"  ✗ FAILED (exit {proc.returncode}, {elapsed/60:.1f}min)")
            log(f"  Last 10 lines of {log_path.name}:")
            tail = proc.stdout.splitlines()[-10:] if proc.stdout else ["(no output)"]
            for line in tail:
                log(f"    {line}")
            return False, elapsed

    elapsed = time.time() - start
    log(f"  ✓ {elapsed/60:.1f}min")

    # Mark stage done on the checkpoints volume for idempotent re-runs.
    try:
        from src.rababa.training.resume import mark_stage_done
        vol_root = Path("/checkpoints")
        if vol_root.is_dir():
            mark_stage_done(vol_root, stage.name, extra={"elapsed_seconds": elapsed})
    except (ImportError, OSError):
        pass  # not on Modal / volume not mounted — silently skip

    return True, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only-lang", choices=["arabic", "hebrew"], default=None,
                        help="Run only one language (default: both)")
    parser.add_argument("--skip-arabic-pro", action="store_true",
                        help="Skip the larger Arabic Pro model (12L/768d) stages")
    parser.add_argument("--skip-to", type=int, default=1,
                        help="Skip ahead to this stage index (1-based). Default: 1")
    parser.add_argument("--only-stage", type=str, default=None,
                        help="Run only the named stage (e.g. 'pretrain_arabic')")
    parser.add_argument("--no-pull", action="store_true",
                        help="Skip the artifact-pull stage")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the benchmark stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--force", action="store_true",
                        help="Re-run stages even if marked done on the volume")
    args = parser.parse_args(argv)

    if args.force:
        os.environ["RABABA_FORCE"] = "1"

    skip_hebrew = args.only_lang == "arabic"
    only_arabic = args.only_lang == "arabic"
    only_hebrew = args.only_lang == "hebrew"

    # Build all relevant stages, then filter post-hoc for --only-lang.
    stages = build_stages(
        skip_hebrew=skip_hebrew,  # If only_hebrew, keep Hebrew stages; we filter Arabic below.
        skip_arabic_pro=args.skip_arabic_pro,
        no_pull=args.no_pull,
        no_benchmark=args.no_benchmark,
    )
    if only_hebrew:
        # Drop Arabic-only stages (fetch_arabic, pretrain_arabic, etc.) plus
        # the Arabic benchmark. Keep Hebrew stages + cross-cutting (pull).
        stages = [
            s for s in stages
            if "arabic" not in s.name or s.name == "pull"
        ]
        # Renumber so logging is clean.
        for i, s in enumerate(stages, start=1):
            s.index = i

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    summary_path = run_dir / "summary.json"

    total_estimate_min = sum(s.estimated_minutes for s in stages)

    log(f"Run dir: {run_dir}")
    log(f"Log: {log_path}")
    log(f"Stages: {len(stages)}")
    log(f"Estimated total: ~{total_estimate_min // 60}h {total_estimate_min % 60}m")
    log(f"{'DRY RUN — no commands will execute' if args.dry_run else 'Live run'}")
    log("")

    summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "dry_run": args.dry_run,
        "stages": [],
        "config": {
            "only_lang": args.only_lang,
            "skip_to": args.skip_to,
            "only_stage": args.only_stage,
            "no_pull": args.no_pull,
            "no_benchmark": args.no_benchmark,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    for stage in stages:
        # Apply filters
        if args.only_stage and stage.name != args.only_stage:
            log(f"[{stage.index}/{len(stages)}] {stage.name}: skipped (only-stage filter)")
            continue
        if stage.index < args.skip_to:
            log(f"[{stage.index}/{len(stages)}] {stage.name}: skipped (skip-to {args.skip_to})")
            continue

        log(f"[{stage.index}/{len(stages)}] {stage.name} — {stage.description}")

        success, elapsed = execute_stage(stage, log_path, dry_run=args.dry_run)

        summary["stages"].append({
            **stage.as_dict(),
            "success": success if not args.dry_run else True,
            "elapsed_seconds": elapsed,
        })
        summary_path.write_text(json.dumps(summary, indent=2))

        if not success:
            log("")
            log(f"FAILED at stage {stage.index} ({stage.name})")
            log(f"Resume with: python scripts/train_all.py --skip-to {stage.index}")
            log(f"Full log: {log_path}")
            return 1

    summary["finished_at"] = datetime.now().isoformat()
    summary_path.write_text(json.dumps(summary, indent=2))

    log("")
    log(f"DONE — summary written to {summary_path}")

    if not args.dry_run and not args.no_benchmark:
        arabic_benchmark = run_dir / "benchmark-v0.1.0-arabic.json"
        if arabic_benchmark.is_file():
            result = json.loads(arabic_benchmark.read_text())
            log("")
            log("=== Arabic v0.1.0 benchmark (vs legacy 4.52%) ===")
            log(f"  DER:               {result.get('der', 'n/a')}")
            log(f"  Per-ex accuracy:   {result.get('per_example_accuracy', 'n/a')}")
            log(f"  Model size:        {result.get('onnx_size_bytes', 0) / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())

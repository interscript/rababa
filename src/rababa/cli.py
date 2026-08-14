"""CLI entry points — single code path for Arabic and Hebrew.

Task dispatch (dataset, collate) lives in `rababa.tasks`; model dispatch
(single vs multi head) lives in `rababa.models.base.build_model`. These
commands just wire CLI args to those modules.

    rababa-pretrain  --task rababa_arabic_pretrain --data-root ... --out-root ...
    rababa-train     --task rababa_arabic --data-root ... --out-root ...
    rababa-export    --task rababa_arabic --version v0.1.0 --checkpoint ...
    rababa-evaluate  --task rababa_arabic --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_task_config, to_dict
from .evaluate import diacritization_error_rate, per_example_accuracy
from .export import export_student_onnx, quantize_dynamic_int8
from .models.base import build_model
from .tasks import build_mlm_loaders, build_supervised_loaders, build_test_loader
from .training import pretrain_mlm, train_supervised


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, help="Task name (rababa_arabic, rababa_hebrew, *_pretrain)")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=Path("models"))


def train_main(argv: list[str] | None = None) -> int:
    """`rababa-train` — Tier 1 supervised training (Arabic or Hebrew)."""
    p = argparse.ArgumentParser(description="Run supervised training")
    _common_args(p)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--init-from-pretrain", type=Path, default=None,
        help="Path to MLM encoder checkpoint (output of rababa-pretrain)",
    )
    args = p.parse_args(argv)

    cfg = load_task_config(args.task)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.init_from_pretrain is not None:
        cfg.train.init_from_pretrain = str(args.init_from_pretrain)

    train_loader, val_loader = build_supervised_loaders(
        cfg, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    ckpt_root = args.out_root / args.task / "checkpoints"
    train_supervised(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=to_dict(cfg),  # type: ignore[arg-type]
        device=device,
        ckpt_root=ckpt_root,
    )
    print(f"Training complete. Checkpoints in {ckpt_root}")
    print(f"Best checkpoint: {ckpt_root / 'best.pt'}")
    return 0


def pretrain_main(argv: list[str] | None = None) -> int:
    """`rababa-pretrain` — MLM pretraining (Arabic or Hebrew)."""
    p = argparse.ArgumentParser(description="Run MLM pretraining")
    _common_args(p)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)

    cfg = load_task_config(args.task)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    train_loader, val_loader = build_mlm_loaders(
        cfg, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    ckpt_root = args.out_root / args.task / "checkpoints"
    pretrain_mlm(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=to_dict(cfg),  # type: ignore[arg-type]
        device=device,
        ckpt_root=ckpt_root,
    )
    print(f"Pretraining complete. Encoder checkpoint: {ckpt_root / 'best.pt'}")
    return 0


def export_main(argv: list[str] | None = None) -> int:
    """`rababa-export` — export checkpoint → ONNX fp32 + int8 (any task)."""
    p = argparse.ArgumentParser(description="Export model to ONNX")
    _common_args(p)
    p.add_argument("--version", required=True, help="Version string (e.g. v0.1.0)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument(
        "--format", choices=["onnx", "tflite"], default="onnx",
        help="Output format: onnx (default) or tflite (for LiteRT.js)",
    )
    args = p.parse_args(argv)

    cfg = load_task_config(args.task)
    cfg_dict = to_dict(cfg)  # type: ignore[arg-type]
    batch_size = cfg.model.get("batch_size", 32)
    max_len = cfg.model.get("max_len", 200)

    out_dir = args.out_root / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "tflite":
        from .export_tflite import export_student_tflite
        tflite_path = out_dir / f"{args.task}-{args.version}-fp32.tflite"
        export_student_tflite(args.checkpoint, cfg_dict, tflite_path, batch_size, max_len)
        print(f"Exported {args.task} {args.version} (TFLite) → {tflite_path}")
        return 0

    fp32_path = out_dir / f"{args.task}-{args.version}-fp32.onnx"
    export_student_onnx(args.checkpoint, cfg_dict, fp32_path, batch_size, max_len)

    if not args.no_quantize:
        q8_path = out_dir / f"{args.task}-{args.version}-q8.onnx"
        quantize_dynamic_int8(fp32_path, q8_path)

    print(f"Exported {args.task} {args.version} (ONNX) → {out_dir}")
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    """`rababa-evaluate` — DER + per-example accuracy on test split (any task)."""
    p = argparse.ArgumentParser(description="Evaluate model")
    _common_args(p)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    cfg = load_task_config(args.task)
    cfg_dict = to_dict(cfg)  # type: ignore[arg-type]
    device = torch.device(args.device)

    model = build_model(cfg_dict).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    head_names = model.head_names()
    loader = build_test_loader(task=args.task, batch_size=args.batch_size)

    head_der = [0.0] * len(head_names)
    head_acc = [0.0] * len(head_names)
    aggregate_wrong = 0
    aggregate_total = 0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            targets = [t.to(device) for t in batch.targets]
            outputs = model.forward_heads(src, lengths)
            any_wrong = None
            any_evaluable = None
            for h_idx, (logits, target) in enumerate(zip(outputs, targets, strict=True)):
                head_der[h_idx] += diacritization_error_rate(logits, target) * src.size(0)
                head_acc[h_idx] += per_example_accuracy(logits, target) * src.size(0)
                preds = logits.argmax(dim=-1)
                head_mask = target != 0
                head_wrong = (preds != target) & head_mask
                any_wrong = head_wrong if any_wrong is None else (any_wrong | head_wrong)
                any_evaluable = head_mask if any_evaluable is None else (any_evaluable | head_mask)
            aggregate_wrong += any_wrong.sum().item()
            aggregate_total += any_evaluable.sum().item()
            total_n += src.size(0)

    result = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "head_names": head_names,
        "n_examples": total_n,
        "per_head_der": [d / max(1, total_n) for d in head_der],
        "per_head_per_example_accuracy": [a / max(1, total_n) for a in head_acc],
        "der_aggregate": aggregate_wrong / max(1, aggregate_total),
        "der": aggregate_wrong / max(1, aggregate_total),
        "per_example_accuracy": head_acc[0] / max(1, total_n),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

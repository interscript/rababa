"""Benchmark an ONNX diacritization model on a test split.

Used to compare legacy (2021) and modern (post-2024) models on the same
test set, so we can verify "must not regress" before shipping.

Handles both single-head (Arabic: 1 output) and multi-head (Hebrew:
3 outputs) ONNX contracts. For multi-head, reports per-head DER plus
an aggregate "any head wrong" DER.

Usage:
    python -m rababa.benchmark --onnx models-data/arabic-model.onnx
    python -m rababa.benchmark --onnx models/rababa_arabic-v0.1.0-q8.onnx \\
        --output benchmark-v0.1.0.json
    python -m rababa.benchmark --onnx models/rababa_hebrew-v0.1.0-q8.onnx \\
        --task rababa_hebrew
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from .datasets import load_nakdimon, load_tashkeela
from .evaluate import diacritization_error_rate, per_example_accuracy
from .training.collate import Batch, collate_batch, multi_head_collate_batch


ARABIC_TASKS = {"rababa_arabic", "rababa_arabic_pretrain"}
HEBREW_TASKS = {"rababa_hebrew", "rababa_hebrew_pretrain"}


def _is_hebrew_task(task: str) -> bool:
    return task in HEBREW_TASKS or "hebrew" in task


def _detect_io_contract(sess: ort.InferenceSession) -> dict[str, Any]:
    """Inspect ONNX inputs/outputs and return the I/O contract."""
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    return {
        "input_names": [i.name for i in inputs],
        "input_shapes": [i.shape for i in inputs],
        "output_names": [o.name for o in outputs],
        "output_shapes": [o.shape for o in outputs],
        "has_lengths_input": len(inputs) > 1 and inputs[1].name == "lengths",
        "fixed_batch_size": inputs[0].shape[0] if isinstance(inputs[0].shape[0], int) else None,
    }


def _load_test_loader(
    task: str,
    split: str,
    cleaner: str | None,
    batch_size: int,
    max_len: int,
    limit: int | None,
    fixed_batch_size: int | None,
) -> tuple[DataLoader, list[str]]:
    """Build a test DataLoader appropriate for the task. Returns (loader, head_names)."""
    if _is_hebrew_task(task):
        cleaner = cleaner or "hebrew"
        ds = load_nakdimon(split, cleaner=cleaner, max_len=max_len)
        head_names = ["niqqud", "dagesh", "sin"]
        collate = multi_head_collate_batch
    else:
        cleaner = cleaner or "arabic"
        ds = load_tashkeela(split, cleaner=cleaner)
        head_names = ["output"]
        collate = collate_batch

    if limit is not None:
        ds.examples = ds.examples[:limit]

    effective_batch = fixed_batch_size or batch_size
    drop_last = fixed_batch_size is not None
    loader = DataLoader(
        ds,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=drop_last,
    )
    return loader, head_names


def benchmark_onnx(
    onnx_path: Path,
    task: str = "rababa_arabic",
    split: str = "test",
    batch_size: int = 32,
    max_len: int = 200,
    cleaner: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run ONNX inference on a test split, return DER + accuracy metrics.

    Returns a dict with per-head `der` and `per_example_accuracy` plus
    an aggregate `der_aggregate` (fraction of positions where ANY head
    was wrong — the user-visible error rate).
    """
    sess = ort.InferenceSession(str(onnx_path))
    contract = _detect_io_contract(sess)
    onnx_output_names = contract["output_names"]

    loader, dataset_head_names = _load_test_loader(
        task=task,
        split=split,
        cleaner=cleaner,
        batch_size=batch_size,
        max_len=max_len,
        limit=limit,
        fixed_batch_size=contract["fixed_batch_size"],
    )

    input_name = contract["input_names"][0]
    has_lengths = contract["has_lengths_input"]
    fixed_seq_len = None
    if contract["fixed_batch_size"] is not None:
        # Fixed-shape model — pad input to the seq dim the ONNX expects.
        shape = contract["input_shapes"][0]
        if len(shape) > 1 and isinstance(shape[1], int):
            fixed_seq_len = shape[1]

    n_heads = len(onnx_output_names)
    head_der = [0.0] * n_heads
    head_acc = [0.0] * n_heads
    aggregate_wrong = 0
    aggregate_total = 0
    total_n = 0
    n_batches = 0

    for batch in loader:
        src_np = batch.src.numpy()
        lengths_np = batch.lengths.numpy()
        head_targets_np = [t.numpy() for t in batch.targets]

        # Pad to fixed seq_len if model requires it (both src and targets).
        if fixed_seq_len is not None and src_np.shape[1] < fixed_seq_len:
            pad_width = fixed_seq_len - src_np.shape[1]
            src_np = np.concatenate(
                [src_np, np.zeros((src_np.shape[0], pad_width), dtype=src_np.dtype)],
                axis=1,
            )
            head_targets_np = [
                np.concatenate(
                    [t, np.zeros((t.shape[0], pad_width), dtype=t.dtype)],
                    axis=1,
                )
                for t in head_targets_np
            ]

        feed: dict[str, np.ndarray] = {input_name: src_np}
        if has_lengths:
            feed["lengths"] = lengths_np

        outputs = sess.run(None, feed)
        head_logits = [torch.from_numpy(o) for o in outputs]
        head_targets = [torch.from_numpy(t) for t in head_targets_np]

        if len(head_logits) != len(head_targets):
            raise ValueError(
                f"ONNX has {len(head_logits)} outputs but dataset produces "
                f"{len(head_targets)} targets — task/dataset mismatch"
            )

        # Track per-position "any head wrong" for aggregate DER.
        any_wrong: torch.Tensor | None = None
        any_evaluable: torch.Tensor | None = None

        for h_idx, (logits, target) in enumerate(zip(head_logits, head_targets, strict=True)):
            head_der[h_idx] += diacritization_error_rate(logits, target) * src_np.shape[0]
            head_acc[h_idx] += per_example_accuracy(logits, target) * src_np.shape[0]
            preds = logits.argmax(dim=-1)
            head_mask = target != 0  # PAD_ID = 0
            head_wrong = (preds != target) & head_mask
            any_wrong = head_wrong if any_wrong is None else (any_wrong | head_wrong)
            any_evaluable = head_mask if any_evaluable is None else (any_evaluable | head_mask)

        aggregate_wrong += any_wrong.sum().item()
        aggregate_total += any_evaluable.sum().item()
        total_n += src_np.shape[0]
        n_batches += 1

    der_aggregate = aggregate_wrong / max(1, aggregate_total)

    return {
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "task": task,
        "split": split,
        "cleaner": cleaner or ("hebrew" if _is_hebrew_task(task) else "arabic"),
        "n_examples": total_n,
        "n_batches": n_batches,
        "head_names": onnx_output_names,
        "per_head_der": [d / max(1, total_n) for d in head_der],
        "per_head_per_example_accuracy": [a / max(1, total_n) for a in head_acc],
        # Aggregate DER = fraction of evaluable positions where ANY head was wrong.
        # For single-head models this equals per_head_der[0].
        "der_aggregate": der_aggregate,
        # Convenience aliases for the most common single-number comparisons.
        "der": der_aggregate,
        "per_example_accuracy": head_acc[0] / max(1, total_n) if head_acc else 0.0,
        "io_contract": contract,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark an ONNX diacritization model")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--task", default="rababa_arabic")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--cleaner", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only first N examples (smoke check)")
    p.add_argument("--output", type=Path, default=None,
                   help="Write JSON result to this path")
    args = p.parse_args(argv)

    result = benchmark_onnx(
        onnx_path=args.onnx,
        task=args.task,
        split=args.split,
        batch_size=args.batch_size,
        max_len=args.max_len,
        cleaner=args.cleaner,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

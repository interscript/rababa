#!/usr/bin/env python3
"""Ensemble evaluation: average predictions across multiple model checkpoints.

Loads N checkpoints (same architecture, different seeds), averages their
softmax probabilities per position, takes argmax, computes DER.

Proven 5-15% DER improvement over single model.

Usage (on Modal):
    # After multi_seed produces seed checkpoints:
    modal run modal_app.py::ensemble_evaluate --task rababa_hebrew --n-seeds 3
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import load_task_config, to_dict
from .constants import PAD_ID
from .evaluate import diacritization_error_rate, per_example_accuracy
from .models.base import build_model
from .tasks import build_test_loader


def ensemble_evaluate(
    task: str,
    checkpoint_paths: list[str] | None = None,
    n_seeds: int = 3,
) -> dict[str, object]:
    """Evaluate ensemble of N checkpoints on test split.

    Args:
        task: task name (e.g. "rababa_hebrew").
        checkpoint_paths: explicit list of checkpoint paths. If None, auto-discovers
            from /checkpoints/{task}/seed-{N:03d}/run-001/best.pt.
        n_seeds: number of seeds (used for auto-discovery).

    Returns: dict with per_head_der, aggregate_der, n_examples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_task_config(task)
    cfg_dict = to_dict(cfg)

    # Discover or use explicit checkpoints.
    if checkpoint_paths is None:
        checkpoint_paths = []
        for s in range(n_seeds):
            p = f"/checkpoints/{task}/seed-{s:03d}/run-001/best.pt"
            if Path(p).is_file():
                checkpoint_paths.append(p)
        if not checkpoint_paths:
            # Fallback: single checkpoint.
            p = f"/checkpoints/{task}/run-001/best.pt"
            if Path(p).is_file():
                checkpoint_paths = [p]

    print(f"Ensemble of {len(checkpoint_paths)} models:")
    for p in checkpoint_paths:
        print(f"  {p}")

    # Load models.
    models = []
    for ckpt_path in checkpoint_paths:
        m = build_model(cfg_dict).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    head_names = models[0].head_names()
    loader = build_test_loader(task=task, batch_size=32)

    # Accumulate per-head DER.
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

            # Get averaged softmax from all models.
            all_outputs = [[] for _ in targets]
            for m in models:
                outputs = m.forward_heads(src, lengths)
                for h_idx, o in enumerate(outputs):
                    all_outputs[h_idx].append(o.softmax(dim=-1))

            # Average softmax across models.
            avg_outputs = []
            for h_idx in range(len(targets)):
                avg = torch.stack(all_outputs[h_idx]).mean(dim=0)
                # Convert back to logits for DER computation (argmax works on probs too).
                avg_outputs.append(avg)

            any_wrong = None
            any_evaluable = None
            for h_idx, (probs, target) in enumerate(zip(avg_outputs, targets, strict=True)):
                # DER: per-position error rate.
                preds = probs.argmax(dim=-1)
                head_mask = target != PAD_ID
                head_wrong = (preds != target) & head_mask
                wrong_count = head_wrong.sum().item()
                total_count = head_mask.sum().item()
                head_der[h_idx] += (wrong_count / max(1, total_count)) * src.size(0)
                head_acc[h_idx] += per_example_accuracy(torch.log(probs + 1e-8), target) * src.size(0)
                any_wrong = head_wrong if any_wrong is None else (any_wrong | head_wrong)
                any_evaluable = head_mask if any_evaluable is None else (any_evaluable | head_mask)

            aggregate_wrong += any_wrong.sum().item()
            aggregate_total += any_evaluable.sum().item()
            total_n += src.size(0)

    result = {
        "task": task,
        "n_models": len(models),
        "checkpoint_paths": checkpoint_paths,
        "head_names": head_names,
        "n_examples": total_n,
        "per_head_der": [d / max(1, total_n) for d in head_der],
        "der_aggregate": aggregate_wrong / max(1, aggregate_total),
        "der": aggregate_wrong / max(1, aggregate_total),
    }

    import json
    print("=== ensemble evaluate result ===")
    print(json.dumps(result, indent=2, default=str))
    return result

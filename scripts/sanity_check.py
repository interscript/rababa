"""Sanity check for trained checkpoints.

Quick checks that catch mode collapse / NaN / regression:
  - All weights finite
  - Forward pass produces finite outputs
  - Outputs differ across different inputs (NOT mode collapse)
  - Predictions have variance (NOT constant)

Usage:
    python scripts/sanity_check.py --task rababa_hebrew \\
        --checkpoint /path/to/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    args = ap.parse_args()

    if not Path(args.checkpoint).is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    if args.task.startswith("rababa"):
        return _check_rababa(args)
    if args.task.startswith("secryst"):
        return _check_secryst(args)
    print(f"ERROR: unknown task prefix: {args.task}")
    return 1


def _check_rababa(args) -> int:
    from rababa.config import load_task_config, to_dict
    from rababa.models.base import build_model

    cfg = load_task_config(args.task)
    cfg_dict = to_dict(cfg)
    model = build_model(cfg_dict)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    # Check 1: all weights finite.
    n_nan = sum(int(not torch.isfinite(p).all().item()) for p in model.parameters())
    print(f"[check] NaN/Inf params: {n_nan}")
    if n_nan > 0:
        print("FAIL: model has NaN/Inf params")
        return 1

    # Check 2: forward produces finite outputs.
    if args.task.startswith("rababa_hebrew"):
        from rababa.constants_hebrew import INPUT_VOCAB_SIZE as V
    else:
        from rababa.constants import INPUT_VOCAB_SIZE as V
    src1 = torch.randint(1, V - 1, (4, 32))
    src2 = torch.randint(1, V - 1, (4, 32))
    lengths = torch.tensor([32, 32, 32, 32])

    with torch.no_grad():
        out1 = model.forward_heads(src1, lengths) if hasattr(model, "forward_heads") else [model(src1, lengths)]
        out2 = model.forward_heads(src2, lengths) if hasattr(model, "forward_heads") else [model(src2, lengths)]

    for i, (o1, o2) in enumerate(zip(out1, out2)):
        finite = torch.isfinite(o1).all().item() and torch.isfinite(o2).all().item()
        print(f"[check] head {i}: shape={tuple(o1.shape)} finite={finite}")
        if not finite:
            print(f"FAIL: head {i} has non-finite outputs")
            return 1

    # Check 3: outputs differ across different inputs (mode-collapse detection).
    # Compare argmax predictions on src1 vs src2.
    pred1 = [o1.argmax(dim=-1) for o1 in out1]
    pred2 = [o2.argmax(dim=-1) for o2 in out2]
    for i, (p1, p2) in enumerate(zip(pred1, pred2)):
        same = (p1 == p2).float().mean().item()
        print(f"[check] head {i}: prediction agreement across diff inputs = {same:.3f}")
        if same > 0.95:
            print(f"WARN: head {i} may be mode-collapsed (predictions match >95% across different inputs)")

    # Check 4: predictions have variance per-input (not constant within a sequence).
    for i, p1 in enumerate(pred1):
        unique_vals = p1.unique().numel()
        total_vals = p1.numel()
        print(f"[check] head {i}: {unique_vals}/{total_vals} unique predicted values")
        if unique_vals < 4:
            print(f"WARN: head {i} predictions are nearly constant")

    print("OK")
    return 0


def _check_secryst(args) -> int:
    from secryst.config import load_task_config, to_dict
    from secryst.models.base import build_model

    cfg = load_task_config(args.task)
    cfg_dict = to_dict(cfg)
    model = build_model(cfg_dict)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    # Check 1.
    n_nan = sum(int(not torch.isfinite(p).all().item()) for p in model.parameters())
    print(f"[check] NaN/Inf params: {n_nan}")
    if n_nan > 0:
        print("FAIL: model has NaN/Inf params")
        return 1

    # Check 2: greedy decode on different inputs produces different outputs.
    from secryst.constants import BOS_ID, EOS_ID, PAD_ID
    V = model.encoder.embedding.num_embeddings
    src1 = torch.randint(4, V - 1, (4, 16))
    src2 = torch.randint(4, V - 1, (4, 16))
    src1[:, 0] = 4  # avoid BOS/EOS/PAD/UNK region
    src2[:, 0] = 4
    lengths = torch.tensor([16, 16, 16, 16])

    from secryst.decoding.beam import greedy_decode
    preds1 = greedy_decode(model, src1, lengths, max_len=32)
    preds2 = greedy_decode(model, src2, lengths, max_len=32)

    print(f"[check] sample preds (input batch 1): {preds1[:2]}")
    print(f"[check] sample preds (input batch 2): {preds2[:2]}")

    n_diff = sum(1 for a, b in zip(preds1, preds2) if a != b)
    print(f"[check] {n_diff}/4 predictions differ across two different inputs")
    if n_diff == 0:
        print("FAIL: all predictions identical across different inputs = mode collapse")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

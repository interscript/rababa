#!/usr/bin/env python3
"""Extract ResFormer λ values from a checkpoint.

After training, inspect what the model learned for the value-residual
mixture coefficients. Paper Fig. 6 shows later layers learn larger λ_1
(deeper dependence on V_1). If our model follows this pattern, ResFormer
is being used correctly.

Usage:
    python scripts/inspect_resformer_lambdas.py /tmp/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Path to .pt state_dict file")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        print(f"Error: {args.checkpoint} not found", file=sys.stderr)
        return 2

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # State may be a raw state_dict or wrapped in {"model": ...}.
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    found_any = False
    print(f"\nResFormer λ values in {args.checkpoint.name}:\n")
    print(f"  {'layer':<35} {'λ_1':>10} {'λ_2':>10} {'λ_1/λ_2':>10}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")

    for key in sorted(state.keys()):
        if "resformer_lambda" in key:
            found_any = True
            # Layer path looks like: layers.{i}.resformer_lambda1
            layer_path = key.rsplit(".", 1)[0]
            lam_type = key.rsplit(".", 1)[1]
            value = state[key].item() if torch.is_tensor(state[key]) else float(state[key])

            # Cache values per layer path for ratio computation.
            if not hasattr(main, "_lams"):
                main._lams = {}
            main._lams.setdefault(layer_path, {})[lam_type] = value

    # Print rows grouped by layer.
    for layer_path in sorted(main._lams.keys()):
        lams = main._lams[layer_path]
        lam1 = lams.get("resformer_lambda1", 0.0)
        lam2 = lams.get("resformer_lambda2", 0.0)
        ratio = lam1 / lam2 if lam2 != 0 else float("inf")
        print(f"  {layer_path:<35} {lam1:>10.4f} {lam2:>10.4f} {ratio:>10.4f}")

    main._lams = {}  # reset for next invocation

    if not found_any:
        print("  (no resformer_lambda parameters found — model wasn't trained with ResFormer)")
        return 1

    print("\nInterpretation:")
    print("  - λ_1/λ_2 > 1: layer relies more on V_1 (first-layer value) than V_n.")
    print("    Paper Fig. 6 shows later layers learn larger λ_1, validating")
    print("    that deep layers benefit most from first-layer token-level info.")
    print("  - λ_1/λ_2 < 1: layer relies more on its own V_n. ResFormer residual")
    print("    is essentially inactive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Per-Head Muon — K3 SOTA optimizer upgrade.

Standard Muon orthogonalizes the entire QKV weight matrix as one block.
Per-Head Muon (Kimi K3, arXiv:2607.24653) treats each attention head's
slice of Q and K (and out_proj's per-head slice) as an independent
matrix for Newton-Schulz orthogonalization.

Why this matters: attention heads learn different things. Orthogonalizing
the whole QKV matrix couples them via the NS iteration. Per-Head NS
preserves each head's individual geometry, giving more principled updates.

Implementation: when the optimizer sees a parameter whose name matches
the per-head pattern (qkv, qkv_self, out_proj, out_self, q_cross, etc.),
it reshapes the matrix to per-head slices, runs NS per slice, and
reshapes back.

Detection is name-based (not architecture-aware) so this stays
optimizer-only — no model code changes needed. Open/closed: existing
Muon class unchanged, this is a new class that wraps it.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from .optim import Muon, zeropower_via_newtonschulz5


# Param name patterns that signal "this is a per-head attention weight".
# Each such matrix has shape (heads * head_dim, dim) or (dim, heads * head_dim)
# and can be reshaped to (heads, head_dim, dim) for per-head NS.
PER_HEAD_PATTERNS = ("qkv", "out_proj", "out_self", "out_cross", "q_cross", "kv_cross")


def _is_per_head_param(name: str, p: nn.Parameter) -> bool:
    """True if this param should use per-head NS."""
    if p.ndim != 2:
        return False
    return any(pat in name for pat in PER_HEAD_PATTERNS)


def _infer_head_count(name: str, p: nn.Parameter) -> int | None:
    """Infer the number of heads from the param shape.

    For fused QKV weight (3*dim, dim): the first axis is 3 * heads * head_dim.
    For per-head linear (q_cross: dim, dim): the first axis is heads * head_dim.
    For out_proj (dim, dim): second axis is heads * head_dim.

    We don't know `heads` from the shape alone — we infer it by factoring
    the larger axis into (heads, head_dim) such that head_dim divides
    evenly. Caller can override via `heads_hint` if needed.
    """
    rows, cols = p.shape
    # Try common head counts (8, 12, 6, 4 — typical Transformer configs).
    for candidate in (8, 12, 6, 4, 16, 2):
        if rows % candidate == 0 and cols % candidate == 0:
            # Prefer the smaller head_dim for stability.
            row_head_dim = rows // candidate
            col_head_dim = cols // candidate
            if row_head_dim >= 8 and col_head_dim >= 8:
                return candidate
    return None


def per_head_newton_schulz(
    G: torch.Tensor,
    name: str,
    heads: int | None,
    steps: int = 5,
) -> torch.Tensor:
    """Run NS on per-head slices of an attention weight.

    Reshapes G into per-head slices, runs NS on each, reshapes back.
    Falls back to standard whole-matrix NS if head count can't be inferred
    or matrix doesn't factor cleanly.
    """
    if heads is None:
        heads = _infer_head_count(name, G)
    if heads is None or heads < 2:
        return zeropower_via_newtonschulz5(G, steps=steps)

    rows, cols = G.shape
    # Fused QKV: rows = 3*heads*head_dim. Factor as (3, heads, head_dim, cols).
    # Detect fused QKV by name.
    if "qkv" in name and rows % (3 * heads) == 0:
        head_dim = rows // (3 * heads)
        # Reshape: (3*heads*head_dim, cols) → (3, heads, head_dim, cols)
        G_reshaped = G.view(3, heads, head_dim, cols)
        # Run NS per (head, qkv_slot) slice.
        out = torch.empty_like(G_reshaped)
        for qkv_idx in range(3):
            for h in range(heads):
                slice_2d = G_reshaped[qkv_idx, h]  # (head_dim, cols)
                out[qkv_idx, h] = zeropower_via_newtonschulz5(slice_2d, steps=steps)
        return out.view(rows, cols)

    # Non-QKV attention weight (out_proj, q_cross, kv_cross, etc.)
    # Factor rows OR cols as (heads, head_dim).
    if rows % heads == 0:
        head_dim = rows // heads
        # Reshape (heads*head_dim, cols) → (heads, head_dim, cols)
        G_reshaped = G.view(heads, head_dim, cols)
        out = torch.empty_like(G_reshaped)
        for h in range(heads):
            out[h] = zeropower_via_newtonschulz5(G_reshaped[h], steps=steps)
        return out.view(rows, cols)
    if cols % heads == 0:
        head_dim = cols // heads
        G_reshaped = G.view(rows, heads, head_dim).transpose(0, 1)  # (heads, rows, head_dim)
        out = torch.empty_like(G_reshaped)
        for h in range(heads):
            out[h] = zeropower_via_newtonschulz5(G_reshaped[h], steps=steps)
        return out.transpose(0, 1).view(rows, cols)

    # Couldn't factor — fall back.
    return zeropower_via_newtonschulz5(G, steps=steps)


class PerHeadMuon(Muon):
    """Muon variant that orthogonalizes per-head slices of attention weights.

    Drop-in replacement for `Muon`. Same constructor signature. Routes
    per-head params (detected by name) through `per_head_newton_schulz`;
    all other 2D params use standard NS.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        heads_hint: int | None = None,
    ) -> None:
        super().__init__(params, lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        self.heads_hint = heads_hint

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                if wd > 0:
                    buf.add_(p, alpha=wd)
                if g.ndim == 2 and min(g.shape) >= 2:
                    # Look up the param name via the optimizer's param-to-name map.
                    name = self._param_names.get(id(p), "")
                    if _is_per_head_param(name, g):
                        update = per_head_newton_schulz(buf, name, self.heads_hint, steps=ns_steps)
                    else:
                        update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    if not torch.isfinite(update).all():
                        continue
                    scale = max(1.0, math.sqrt(max(g.shape) / min(g.shape)))
                    p.add_(update, alpha=-lr * scale)
                else:
                    p.add_(buf, alpha=-lr)
        return loss

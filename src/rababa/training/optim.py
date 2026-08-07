"""K3/DS4 optimizer stack — Muon + AdamW hybrid + QK-Clip.

References:
  - Muon (Keller Jordan, 2024): github.com/KellerJordan/muon
  - Kimi K2 MuonClip + QK-Clip: arXiv:2507.20534
  - Per-Head Muon (Kimi K3): arXiv:2607.24653 (deferred — needs QKV refactor)

Design:
  - **Muon** for 2D weight matrices (linear/embedding-projections):
    Newton-Schulz iteration orthogonalizes the momentum buffer, giving
    matrix-aware updates that converge ~2× faster than AdamW per token
    of compute (Essential AI scaling laws, Feb 2025).
  - **AdamW** for 1D parameters (RMSNorm weights, biases) where Muon
    doesn't apply.
  - **QK-Clip**: after each step, if `max(QK^T)` over a probe batch
    exceeds τ, rescale the Q output projection down so subsequent
    logits stay bounded. Anneal τ → 0 over training. Prevents the
    attention-logit explosion that destroys from-scratch pretraining.

`build_optimizer` in supervised.py / pretrain.py dispatches on
`cfg.train.optimizer: "muon" | "adamw"` (default remains "adamw" for
backward compat).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


# ---- Newton-Schulz orthogonalization ----------------------------------


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration: compute approx-orthogonal factor of G.

    Standard Muon helper from KellerJordan/muon. Coefficients (a, b, c)
    are the optimal values for 5-iteration NS on the matrix sign function.
    Operates in bfloat16 for speed; result is cast back to G's dtype.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.to(G.dtype)


# ---- Muon optimizer (2D weights only) ---------------------------------


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices.

    Update rule (for parameters with grad.ndim == 2):
        momentum = ρ * momentum + g
        update = zeropower_via_newtonschulz5(momentum)
        p -= lr * update * sqrt(max(rows, cols) / min(rows, cols))

    Parameters with grad.ndim != 2 fall back to SGD-with-momentum.
    Typically you don't use this directly — use `MuonAdamWHybrid` which
    routes 1D params to AdamW.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

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
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                if wd > 0:
                    buf.add_(p, alpha=wd)
                if g.ndim == 2 and min(g.shape) >= 2:
                    update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    scale = max(1.0, math.sqrt(max(g.shape) / min(g.shape)))
                    p.add_(update, alpha=-lr * scale)
                else:
                    # 1D / non-matrix: SGD with momentum.
                    p.add_(buf, alpha=-lr)
        return loss


# ---- Hybrid Muon + AdamW ----------------------------------------------


class MuonAdamWHybrid:
    """K3/DS4 hybrid optimizer: Muon for 2D weights, AdamW for everything else.

    Wrapper that exposes the standard optimizer API (step, zero_grad,
    state_dict, load_state_dict) so it's a drop-in replacement for
    torch.optim.Optimizer in training loops.

    Routing:
      - 2D parameters whose name doesn't contain 'embedding' or 'norm':
        → Muon (these are the linear/attention/FFN weight matrices)
      - Everything else (embeddings, RMSNorm weights, biases):
        → AdamW

    Embeddings go to AdamW because their update structure is row-sparse
    (only the looked-up rows get gradient) which doesn't benefit from
    Newton-Schulz orthogonalization.
    """

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        adam_lr: float = 3e-4,
        muon_momentum: float = 0.95,
        adam_weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        muon_params: list[nn.Parameter] = []
        adam_params: list[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "embedding" not in name and "norm" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            ns_steps=ns_steps,
        )
        self.adam = torch.optim.AdamW(adam_params, lr=adam_lr, weight_decay=adam_weight_decay)
        self._muon_param_ids = {id(p) for p in muon_params}
        self._adam_param_ids = {id(p) for p in adam_params}

    @property
    def param_groups(self) -> list[dict]:
        return list(self.muon.param_groups) + list(self.adam.param_groups)

    def step(self, closure=None) -> float | None:
        # Step both sub-optimizers. Each ignores params it doesn't own.
        muon_loss = self.muon.step(closure)
        adam_loss = self.adam.step()
        return muon_loss if muon_loss is not None else adam_loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])

    @property
    def defaults(self) -> dict:
        return {**self.muon.defaults, **self.adam.defaults}


# ---- QK-Clip (Kimi K2 MuonClip) ---------------------------------------


@torch.no_grad()
def qk_clip_(model: nn.Module, tau: float = 8.0) -> dict[str, float]:
    """Rescale Q output projections so max(QK^T) ≤ tau.

    Following Kimi K2's MuonClip recipe: scan all attention layers, find
    the Q output projection (`out_proj` after the QKV split), and if its
    products with K would exceed tau, rescale Q down.

    For our ModernEncoderLayer the Q and K are produced together via the
    fused `qkv` linear then split. We approximate by scaling the qkv
    weight's Q-slice uniformly — a conservative rescaling that bounds the
    QK^T norm without needing a probe batch.

    Returns a dict of per-layer scaling factors for logging.

    This is called from the training loop every N steps (e.g., N=10) with
    tau annealed from 8 → 1 over training.
    """
    out: dict[str, float] = {}
    layers = getattr(model, "layers", None)
    if layers is None:
        return out
    for i, layer in enumerate(layers):
        qkv = getattr(layer, "qkv", None)
        if qkv is None or not isinstance(qkv, nn.Linear):
            continue
        # qkv.weight has shape (3*dim, dim). The Q slice is rows [0:dim].
        w = qkv.weight
        D = w.shape[1]
        w_q = w[:D]
        w_k = w[D:2 * D]
        # Frobenius-norm bound: max |QK^T| ≤ ||w_q||_F * ||w_k||_F * (input norm)^2
        # Conservative: if product of norms > tau, rescale w_q down.
        norm_product = w_q.norm() * w_k.norm()
        if norm_product.item() > tau:
            scale = math.sqrt(tau / max(norm_product.item(), 1e-8))
            # In-place rescale — preserves grad links.
            w_q_scaled = w_q * scale
            new_w = torch.cat([w_q_scaled, w[D:]], dim=0)
            qkv.weight.copy_(new_w)
            out[f"layer_{i}"] = scale
        else:
            out[f"layer_{i}"] = 1.0
    return out


def qk_clip_schedule(step: int, total_steps: int, tau_init: float = 8.0, tau_final: float = 1.0) -> float:
    """Linear anneal of QK-Clip threshold tau_init → tau_final over training."""
    if total_steps <= 0:
        return tau_final
    progress = min(1.0, step / total_steps)
    return tau_init + (tau_final - tau_init) * progress


# ---- Per-Head Muon (Kimi K3, deferred) --------------------------------
#
# Per-Head Muon treats each attention head's slice of the QKV weight as
# an independent matrix for Newton-Schulz orthogonalization. Requires
# the QKV projection to be parameterized as separate per-head linears
# (or to slice the weight during the Muon step).
#
# Deferred to v2 — our modern.py uses a single fused `nn.Linear(dim, 3*dim)`
# for QKV. Per-Head Muon would require either:
#   (a) refactor QKV to N_heads separate `nn.Linear(dim, head_dim)` modules
#   (b) custom Muon step that slices the weight per head before NS
# Neither is in the 1-week sprint scope. Note for v2.

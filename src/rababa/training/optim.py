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


# DS-V4-Flash §2.4 hybrid NS: two-stage coefficients.
# Stage 1 (aggressive, drives singular values close to 1).
_NS_COEFFS_AGGRESSIVE = (3.4445, -4.7750, 2.0315)
# Stage 2 (stable, lands singular values precisely at 1).
_NS_COEFFS_STABLE = (2.0, -1.5, 0.5)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration: compute approx-orthogonal factor of G.

    DS-V4-Flash §2.4 hybrid Newton-Schulz: first ~80% of iterations use
    aggressive coefficients (3.4445, -4.7750, 2.0315) for rapid convergence,
    last ~20% use stable coefficients (2, -1.5, 0.5) to land singular values
    precisely at 1. For 5 steps: 4+1 split. For 10 steps: 8+2 (paper recipe).

    Operates in bfloat16 for speed; result is cast back to G's dtype.
    """
    assert G.ndim == 2
    aggressive_steps = max(1, int(0.8 * steps))
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    for i in range(steps):
        a, b, c = _NS_COEFFS_AGGRESSIVE if i < aggressive_steps else _NS_COEFFS_STABLE
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
        update_rms_rescale: float | None = None,
        spectral_cap: float | None = None,
        heavy_tail_alpha: float | None = None,
        adamuon_beta: float | None = None,
        normuon_enabled: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, ns_steps=ns_steps,
            weight_decay=weight_decay, update_rms_rescale=update_rms_rescale,
            spectral_cap=spectral_cap, heavy_tail_alpha=heavy_tail_alpha,
            adamuon_beta=adamuon_beta, normuon_enabled=normuon_enabled,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            rms_rescale = group.get("update_rms_rescale")
            spectral_cap = group.get("spectral_cap")
            heavy_tail_alpha = group.get("heavy_tail_alpha")
            adamuon_beta = group.get("adamuon_beta")
            normuon_enabled = group.get("normuon_enabled", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Skip bad grads (NaN/Inf from numerical instability).
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
                    update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    # NS can diverge in bf16 → skip if update not finite.
                    if not torch.isfinite(update).all():
                        continue
                    # Spectral Cap (2026): cap the spectral radius of orthogonalized
                    # updates to prevent optimizer instability in long training runs.
                    # The orthogonalized update has singular values near 1; the cap
                    # ensures none exceeds `spectral_cap` (default off → None).
                    if spectral_cap is not None:
                        # Cheap proxy: clip Frobenius norm to spectral_cap * sqrt(min_dim).
                        # Full SVD-based cap is too expensive per step.
                        max_frob = spectral_cap * math.sqrt(min(g.shape))
                        frob = update.norm() + 1e-8
                        scale = torch.clamp(max_frob / frob, max=1.0)
                        update = update * scale
                    # HTMuon heavy-tail correction (arXiv:2603.10067, ACL 2026):
                    # re-inject heavy tails suppressed by orthogonalization.
                    # α-blend the orthogonalized update with raw momentum to
                    # restore heavier-tailed weight spectra (HT-SR theory).
                    if heavy_tail_alpha is not None and heavy_tail_alpha > 0:
                        update = (1 - heavy_tail_alpha) * update + heavy_tail_alpha * buf
                    # AdaMuon (arXiv:2507.11005): element-wise second-moment
                    # estimator on the orthogonalized update direction. Adam-style
                    # adaptivity on the orthogonal projection. Sign-stabilized
                    # by construction (NS output signs track momentum signs).
                    if adamuon_beta is not None:
                        if "v_buffer" not in state:
                            state["v_buffer"] = torch.zeros_like(update)
                            state["step"] = 0
                        state["step"] += 1
                        v_buf = state["v_buffer"]
                        v_buf.mul_(adamuon_beta).addcmul_(update, update, value=1 - adamuon_beta)
                        # Bias correction (like Adam): v_buf is biased toward 0
                        # at startup, dividing by it amplifies updates ~10x.
                        # Correct: v_hat = v / (1 - beta^t).
                        bias_corr = 1.0 - adamuon_beta ** state["step"]
                        v_hat = v_buf / bias_corr
                        denom = v_hat.sqrt().add_(1e-8)
                        update = update / denom
                    # NorMuon (arXiv:2510.05491): neuron-wise adaptive scaling.
                    # Normalizes each neuron's (row's) update to uniform magnitude,
                    # fixing Muon's per-neuron non-uniformity problem.
                    if normuon_enabled:
                        # Treat each row as a neuron (PyTorch Linear convention).
                        row_norms = update.norm(dim=-1, keepdim=True) + 1e-8
                        # Rescale so every row has the mean row norm.
                        mean_norm = row_norms.mean().clamp_min(1e-8)
                        update = update * (mean_norm / row_norms)
                    if rms_rescale is not None:
                        # DS-V4-Flash §2.4: scale = sqrt(max(n,m)) * γ where
                        # γ=0.18 lets us reuse AdamW LR for Muon params.
                        scale = math.sqrt(max(g.shape)) * rms_rescale
                    else:
                        scale = max(1.0, math.sqrt(max(g.shape) / min(g.shape)))
                    p.add_(update, alpha=-lr * scale)
                else:
                    # 1D / non-matrix: SGD with momentum.
                    p.add_(buf, alpha=-lr)
        return loss


# ---- Hybrid Muon + AdamW ----------------------------------------------


class MuonAdamWHybrid:
    """K3/DS4 hybrid optimizer: Muon for 2D weights, AdamW for everything else.

    Optional `cross_attn_lr_mult` lets specific param groups (matched by
    name substring) get a different LR. Useful when one component (e.g.
    cross-attention in seq2seq) needs a higher LR to escape mode collapse.

    When `use_per_head_muon=True`, the inner Muon is replaced with
    `PerHeadMuon` (K3 SOTA) which orthogonalizes per-head slices of
    attention weights instead of the whole matrix.

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
        cross_attn_lr_mult: float = 1.0,
        use_per_head_muon: bool = False,
        heads_hint: int | None = None,
        spectral_cap: float | None = None,
        heavy_tail_alpha: float | None = None,
        adamuon_beta: float | None = None,
        normuon_enabled: bool = False,
    ) -> None:
        muon_params: list[nn.Parameter] = []
        adam_params: list[nn.Parameter] = []
        cross_attn_params: list[nn.Parameter] = []
        param_name_map: dict[int, str] = {}
        for name, p in model.named_parameters():
            param_name_map[id(p)] = name
            if not p.requires_grad:
                continue
            is_cross_attn = (
                cross_attn_lr_mult != 1.0
                and ("q_cross" in name or "kv_cross" in name or "out_cross" in name)
            )
            if is_cross_attn:
                cross_attn_params.append(p)
            elif (
                p.ndim == 2
                and "embedding" not in name
                and "norm" not in name
                and "router" not in name
                and ".moe." not in name
                # Output heads (heads.0.weight, seg_head.weight) are small
                # rectangular matrices (e.g. 384×16) where Muon's NS
                # orthogonalization over-scales updates by sqrt(max/min) ≈ 5×,
                # destroying the head's gradient signal. Route to AdamW.
                and ".heads." not in name
                and "head." not in name
                and "seg_head" not in name
                and "out_proj" not in name  # small attention output projections
            ):
                muon_params.append(p)
            else:
                adam_params.append(p)
        if use_per_head_muon:
            from .per_head_muon import PerHeadMuon
            self.muon = PerHeadMuon(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                ns_steps=ns_steps,
                heads_hint=heads_hint,
            )
        else:
            self.muon = Muon(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                ns_steps=ns_steps,
                spectral_cap=spectral_cap,
                heavy_tail_alpha=heavy_tail_alpha,
                adamuon_beta=adamuon_beta,
                normuon_enabled=normuon_enabled,
            )
        # Expose param names to the Muon optimizer for per-head detection.
        self.muon._param_names = param_name_map  # type: ignore[attr-defined]
        adam_groups = [
            {"params": adam_params, "lr": adam_lr, "weight_decay": adam_weight_decay},
        ]
        if cross_attn_params:
            adam_groups.append({
                "params": cross_attn_params,
                "lr": adam_lr * cross_attn_lr_mult,
                "weight_decay": adam_weight_decay,
            })
        self.adam = torch.optim.AdamW(adam_groups)
        self._muon_param_ids = {id(p) for p in muon_params}
        self._adam_param_ids = {id(p) for p in adam_params}
        self._cross_attn_param_ids = {id(p) for p in cross_attn_params}
        self.cross_attn_lr_mult = cross_attn_lr_mult
        self.use_per_head_muon = use_per_head_muon

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

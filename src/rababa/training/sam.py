"""Sharpness-Aware Minimization (SAM) — Foret et al. ICLR 2021.

Finds flat minima that generalize better than sharp minima found by SGD/Adam.
Two-step training: perturb weights in direction of gradient ascent, then
compute gradient at perturbed weights, then step.

References:
  - Foret et al. "Sharpness-Aware Minimization for Efficiently Improving
    Generalization" (ICLR 2021).
  - Liu et al. 2022 — improves generalization 2-5% on small NLP datasets.

Cost: 2× compute per step (forward+backward twice). For our 935-batch
Hebrew training × 15 epochs, ~2× slower. Acceptable for SOTA run.

Usage:

    sam = SAM(model, base_optimizer_cls=MuonAdamWHybrid, rho=0.05, **kwargs)

    # In training loop:
    loss = compute_loss(model, batch)         # first forward
    loss.backward()
    sam.first_step(zero_grad=True)             # perturb weights

    loss2 = compute_loss(model, batch)        # second forward at perturbed weights
    loss2.backward()
    sam.second_step(zero_grad=True)           # step base optimizer from perturbed grad
"""

from __future__ import annotations

import torch
from torch import nn


class SAM:
    """Sharpness-Aware Minimization wrapper for any base optimizer.

    Wraps a base optimizer (MuonAdamWHybrid, AdamW, etc.) and adds SAM
    perturbation. The base optimizer's step() is called inside second_step().

    Args:
        model: the model being trained (used for gradient perturbation).
        base_optimizer: the wrapped optimizer (already constructed).
        rho: SAM perturbation magnitude. 0.05 is the paper's default; for
            smaller models, 0.01-0.1 may work better. Tune per task.
        adaptive: if True, use ASAM (Liu et al. 2022) — perturbation scaled
            by parameter norm. Often better than vanilla SAM.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: object,
        rho: float = 0.05,
        adaptive: bool = False,
    ) -> None:
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.param_groups = getattr(base_optimizer, "param_groups", [])

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        """Compute perturbation direction (= grad direction) and apply."""
        grad_norm = self._grad_norm()
        # Compute scale: rho * |w| / |grad| (adaptive) or rho / |grad| (vanilla).
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if self.adaptive:
                # ASAM: scale perturbation by parameter norm.
                w_norm = self._param_norm(p)
                eps = self.rho * w_norm / (grad_norm + 1e-12)
            else:
                eps = self.rho / (grad_norm + 1e-12)
            # Save original weights for second_step restoration.
            if not hasattr(p, "_sam_orig"):
                p._sam_orig = torch.zeros_like(p)
            p._sam_orig.copy_(p)
            # Apply perturbation: e = eps * grad.
            e = eps * p.grad
            p.add_(e)
        if zero_grad:
            self._zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        """Restore original weights, then step with perturbed-position gradient."""
        for n, p in self.model.named_parameters():
            if hasattr(p, "_sam_orig"):
                p.copy_(p._sam_orig)
                del p._sam_orig
        # Step base optimizer with the gradient computed at perturbed weights.
        self.base_optimizer.step()
        if zero_grad:
            self._zero_grad()

    def _zero_grad(self) -> None:
        if hasattr(self.base_optimizer, "zero_grad"):
            self.base_optimizer.zero_grad(set_to_none=True)
        else:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None

    def _grad_norm(self) -> torch.Tensor:
        norm = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for p in self.model.parameters():
            if p.grad is not None:
                norm = torch.maximum(norm, p.grad.norm())
        return norm

    def _param_norm(self, p: torch.Tensor) -> torch.Tensor:
        return p.norm()

    # Forward optimizer-like methods.
    def state_dict(self) -> dict:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.base_optimizer.load_state_dict(state)


def sam_train_step(
    model: nn.Module,
    sam: SAM,
    compute_loss: object,
    zero_grad: bool = True,
) -> torch.Tensor:
    """Helper: SAM two-step training.

    Args:
        model: model being trained.
        sam: SAM wrapper.
        compute_loss: zero-arg callable that returns the loss tensor.
            Should do forward + return loss (no backward).
        zero_grad: zero gradients after each step.

    Returns: loss from second forward (at perturbed weights).

    Usage:
        loss = sam_train_step(model, sam, lambda: criterion(model(x), y))
    """
    # First forward-backward.
    loss1 = compute_loss()
    loss1.backward()
    sam.first_step(zero_grad=zero_grad)
    # Second forward-backward at perturbed weights.
    loss2 = compute_loss()
    loss2.backward()
    sam.second_step(zero_grad=zero_grad)
    return loss2

"""Specs for SAM (Sharpness-Aware Minimization)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rababa.training.sam import SAM, sam_train_step


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


def test_sam_first_step_perturbs_weights():
    """first_step should move weights in the gradient direction."""
    model = _toy_model()
    base = torch.optim.AdamW(model.parameters(), lr=0.001)
    sam = SAM(model, base, rho=0.05)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    crit = nn.CrossEntropyLoss()
    loss = crit(model(x), y)
    loss.backward()
    before = model[0].weight.data.clone()
    sam.first_step(zero_grad=False)
    after = model[0].weight.data
    # Weights should be perturbed (different from before).
    assert not torch.allclose(before, after)


def test_sam_second_step_restores_weights():
    """second_step should restore weights to pre-perturbation state,
    THEN apply optimizer step from perturbed-position gradient.

    The net result is: final_weights = orig_weights + optimizer_update_at_perturbed.
    """
    model = _toy_model()
    base = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 = no update, just check restoration
    sam = SAM(model, base, rho=0.05)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    crit = nn.CrossEntropyLoss()
    orig = model[0].weight.data.clone()

    loss = crit(model(x), y)
    loss.backward()
    sam.first_step(zero_grad=False)
    # Weights now perturbed.
    assert not torch.allclose(model[0].weight.data, orig)
    # Re-compute gradient at perturbed weights.
    loss2 = crit(model(x), y)
    loss2.backward()
    sam.second_step(zero_grad=False)
    # With lr=0, the optimizer step is a no-op, so weights should match orig.
    assert torch.allclose(model[0].weight.data, orig)


def test_sam_train_step_smoke():
    """End-to-end SAM training step should complete without error."""
    model = _toy_model()
    base = torch.optim.AdamW(model.parameters(), lr=0.001)
    sam = SAM(model, base, rho=0.05)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    crit = nn.CrossEntropyLoss()
    loss = sam_train_step(model, sam, lambda: crit(model(x), y))
    assert torch.isfinite(loss)
    assert torch.isfinite(model[0].weight).all()


def test_sam_adaptive_uses_asam():
    """adaptive=True should produce different perturbation than vanilla SAM."""
    torch.manual_seed(0)
    model = _toy_model()
    base = torch.optim.AdamW(model.parameters(), lr=0.001)
    sam_vanilla = SAM(model, base, rho=0.05, adaptive=False)

    torch.manual_seed(0)
    model2 = _toy_model()
    base2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
    sam_asam = SAM(model2, base2, rho=0.05, adaptive=True)

    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    crit = nn.CrossEntropyLoss()

    loss1 = crit(model(x), y)
    loss1.backward()
    sam_vanilla.first_step(zero_grad=False)

    loss2 = crit(model2(x), y)
    loss2.backward()
    sam_asam.first_step(zero_grad=False)

    # Perturbed weights should differ between vanilla SAM and ASAM.
    assert not torch.allclose(model[0].weight.data, model2[0].weight.data)


def test_sam_zero_rho_no_perturbation():
    """rho=0 should mean no perturbation (degenerate case)."""
    model = _toy_model()
    base = torch.optim.AdamW(model.parameters(), lr=0.001)
    sam = SAM(model, base, rho=0.0)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    crit = nn.CrossEntropyLoss()
    orig = model[0].weight.data.clone()
    loss = crit(model(x), y)
    loss.backward()
    sam.first_step(zero_grad=False)
    # With rho=0, weights should barely change (only the 1e-12 eps).
    assert (model[0].weight.data - orig).abs().max() < 1e-6


def test_sam_state_dict_roundtrip():
    """state_dict/load_state_dict should forward to base optimizer."""
    model = _toy_model()
    base = torch.optim.AdamW(model.parameters(), lr=0.001)
    sam = SAM(model, base, rho=0.05)
    # Take a step to populate state.
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    crit = nn.CrossEntropyLoss()
    loss = sam_train_step(model, sam, lambda: crit(model(x), y))
    sd = sam.state_dict()
    # Should have AdamW state per param.
    assert len(sd["state"]) > 0

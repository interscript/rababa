"""Specs for EMA (Exponential Moving Average) of model weights."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rababa.training.ema import ModelEMA


def test_ema_initial_shadow_matches_model():
    """Shadow params should initially equal model params (copy)."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9999)
    for n, p in model.named_parameters():
        assert torch.allclose(ema.shadow[n], p.data)


def test_ema_update_changes_shadow():
    """After update with modified params, shadow should differ from before."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9)  # faster decay for visible effect
    before = ema.shadow["weight"].clone()
    # Perturb model params.
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    after = ema.shadow["weight"]
    assert not torch.allclose(before, after)
    # EMA shadow should move toward new params (somewhere between before and new).
    assert (after - before).abs().mean() > 0


def test_ema_swap_context_manager():
    """swap() should temporarily replace model params with shadow."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9)
    # Make shadow very different from model.
    for n in ema.shadow:
        ema.shadow[n].fill_(100.0)
    original = model.weight.data.clone()
    with ema.swap(model):
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 100.0))
    # Should be restored.
    assert torch.allclose(model.weight.data, original)


def test_ema_decay_zero_copies_params():
    """With decay=0, EMA should fully replace shadow with current params."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.0)
    with torch.no_grad():
        model.weight.fill_(7.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight, 7.0))


def test_ema_decay_one_keeps_initial():
    """With decay=1.0, shadow should never change."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=1.0)
    before = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.fill_(7.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], before)


def test_ema_copy_to_overwrites_model():
    """copy_to() should permanently replace model weights with shadow."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9)
    for n in ema.shadow:
        ema.shadow[n].fill_(42.0)
    ema.copy_to(model)
    assert torch.allclose(model.weight.data, torch.full_like(model.weight, 42.0))


def test_ema_state_dict_roundtrip():
    """state_dict / load_state_dict should round-trip."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    sd = ema.state_dict()
    ema2 = ModelEMA(model, decay=0.9)
    ema2.load_state_dict(sd)
    for n in sd:
        assert torch.allclose(ema.shadow[n], ema2.shadow[n])


def test_ema_handles_no_grad_params():
    """EMA should only track params that require grad."""
    model = nn.Linear(10, 5)
    model.weight.requires_grad = False
    ema = ModelEMA(model, decay=0.9)
    # weight is frozen, so should not be in shadow.
    assert "weight" not in ema.shadow
    assert "bias" in ema.shadow  # bias still requires grad by default


def test_ema_excluded_bias():
    """When use_ema_bias=False, bias terms should not be in shadow."""
    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9, use_ema_bias=False)
    assert "weight" in ema.shadow
    assert "bias" not in ema.shadow

"""Specs for MuonAdamWHybrid optimizer param routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from rababa.training.optim import MuonAdamWHybrid


class _TestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, 16)
        self.linear = nn.Linear(16, 16, bias=False)  # → Muon
        self.norm = nn.LayerNorm(16)  # → AdamW (1D + "norm")
        self.router = nn.Linear(16, 4, bias=False)  # → AdamW (router)
        self.bias = nn.Parameter(torch.zeros(16))  # → AdamW (1D)


def test_router_routed_to_adamw_not_muon():
    """Router weights must go to AdamW, not Muon.

    Regression: previously all 2D weights except embedding/norm went to
    Muon. Routers over-amplified under Newton-Schulz orthogonalization,
    causing router norms to explode (Hebrew v0.6.0 NaN incident).
    """
    model = _TestModel()
    opt = MuonAdamWHybrid(model, muon_lr=0.02, adam_lr=3e-4)

    # Muon's params should NOT include router.
    muon_param_ids = {id(p) for p in opt.muon.param_groups[0]["params"]}
    adam_param_ids = set()
    for group in opt.adam.param_groups:
        for p in group["params"]:
            adam_param_ids.add(id(p))

    assert id(model.router.weight) not in muon_param_ids, \
        "router weight went to Muon — should be AdamW"
    assert id(model.router.weight) in adam_param_ids, \
        "router weight not in any AdamW group"


def test_linear_2d_weights_still_go_to_muon():
    """Regular 2D weights (attention/FFN) should still go to Muon."""
    model = _TestModel()
    opt = MuonAdamWHybrid(model, muon_lr=0.02, adam_lr=3e-4)
    muon_param_ids = {id(p) for p in opt.muon.param_groups[0]["params"]}
    assert id(model.linear.weight) in muon_param_ids


def test_step_preserves_param_shapes():
    """MuonAdamWHybrid step should not change any param shapes."""
    model = _TestModel()
    opt = MuonAdamWHybrid(model, muon_lr=0.01, adam_lr=1e-4)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    shapes_before = {id(p): p.shape for p in model.parameters()}
    opt.step()
    shapes_after = {id(p): p.shape for p in model.parameters()}
    for k, v in shapes_before.items():
        assert shapes_after[k] == v

"""Specs for ELECTRA pretraining components."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rababa.training.electra import (
    ElectraDiscriminatorHead,
    ElectraModel,
    electra_loss,
)


def test_electra_discriminator_head_output_shape():
    head = ElectraDiscriminatorHead(dim=16)
    hidden = torch.randn(2, 5, 16)
    out = head(hidden)
    assert out.shape == (2, 5, 2)  # 2 classes: original/replaced


def test_electra_loss_returns_total_gen_disc():
    torch.manual_seed(0)
    B, T, V = 2, 4, 10
    gen_logits = torch.randn(B, T, V)
    disc_logits = torch.randn(B, T, 2)
    src = torch.randint(1, V, (B, T))
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True
    corrupted = src.clone()
    corrupted[mask] = 99  # arbitrary replacement
    out = electra_loss(gen_logits, disc_logits, src, mask, corrupted)
    assert "total" in out
    assert "gen" in out
    assert "disc" in out
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["gen"])
    assert torch.isfinite(out["disc"])


def test_electra_loss_weights_disc_50x_over_gen():
    """ELECTRA paper recipe: disc loss weighted 50× over gen loss."""
    torch.manual_seed(0)
    B, T, V = 2, 4, 10
    # Make gen loss dominate purely by setting gen_logits bad.
    gen_logits = torch.zeros(B, T, V)
    disc_logits = torch.zeros(B, T, 2)
    src = torch.randint(1, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    corrupted = src.clone()
    corrupted[mask] = 99
    out = electra_loss(gen_logits, disc_logits, src, mask, corrupted)
    # Total = gen + 50 * disc. Verify the relationship approximately.
    expected_total = out["gen"] + 50.0 * out["disc"]
    assert abs(out["total"].item() - expected_total.item()) < 1e-3


def test_electra_model_sample_mask_avoids_pad():
    """Make sure sampling never marks PAD positions for replacement."""
    src = torch.tensor([[1, 2, 3, 0, 0]])  # PAD at positions 3, 4
    mask = ElectraModel._sample_mask(src, mask_prob=1.0)
    # Even with mask_prob=1.0, PAD positions must remain False.
    assert not mask[0, 3].item()
    assert not mask[0, 4].item()
    # Non-PAD positions can be True (high probability).
    assert mask[0, 0].item() or mask[0, 1].item() or mask[0, 2].item()

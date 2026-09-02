"""Specs for distillation loss + teacher averaging."""

from __future__ import annotations

import torch
import torch.nn as nn

from rababa.training.distill import (
    averaged_teacher_logits,
    distillation_loss,
    load_teachers,
)


class _StubHead(nn.Module):
    """Tiny single-head model returning a constant per-position logit."""

    def __init__(self, vocab_size: int = 5, seq_len: int = 4, fill: float = 0.0) -> None:
        super().__init__()
        self.fill = fill
        self.dummy = nn.Linear(1, 1)  # so named_parameters is non-empty

    def forward_heads(self, src, lengths):
        B, T = src.shape
        # Return [logits] — single head.
        logits = torch.full((B, T, 5), self.fill)
        return [logits]

    def head_names(self):
        return ["output"]


def test_distillation_loss_alpha_zero_is_pure_ce():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 4, 5)
    teacher_logits = torch.randn(2, 4, 5)
    target = torch.randint(0, 5, (2, 4))
    target[1, 2:] = 0  # mark pad
    out = distillation_loss(student_logits, teacher_logits, target, alpha=0.0)
    # alpha=0 → loss = CE(student, target)
    ce_expected = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, 5),
        target.reshape(-1),
        ignore_index=0,
    )
    assert abs(out.item() - ce_expected.item()) < 1e-5


def test_distillation_loss_alpha_one_includes_kl():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 4, 5)
    teacher_logits = torch.randn(2, 4, 5)
    target = torch.randint(1, 5, (2, 4))
    out = distillation_loss(student_logits, teacher_logits, target, alpha=1.0)
    assert torch.isfinite(out)
    assert out.item() > 0


def test_averaged_teacher_logits_returns_mean_per_head():
    t1 = _StubHead(fill=2.0)
    t2 = _StubHead(fill=4.0)
    src = torch.zeros((1, 3), dtype=torch.long)
    lengths = torch.tensor([3])
    out = averaged_teacher_logits([t1, t2], src, lengths)
    assert len(out) == 1
    # Mean of 2.0 and 4.0 = 3.0.
    assert torch.allclose(out[0], torch.full((1, 3, 5), 3.0))


def test_load_teachers_loads_state_dicts_and_freezes(tmp_path):
    torch.manual_seed(0)
    # Save a stub model.
    stub = _StubHead()
    stub_path = tmp_path / "teacher.pt"
    torch.save(stub.state_dict(), stub_path)
    # Build cfg dict matching _StubHead's expectation. We bypass
    # build_model by patching the loader.
    import rababa.training.distill as d
    original_build = d.build_model
    d.build_model = lambda cfg: _StubHead()
    try:
        teachers = load_teachers([stub_path], {}, torch.device("cpu"))
    finally:
        d.build_model = original_build
    assert len(teachers) == 1
    for p in teachers[0].parameters():
        assert not p.requires_grad

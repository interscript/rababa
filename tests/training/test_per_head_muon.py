"""Specs for Per-Head Muon (K3 SOTA)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rababa.training.per_head_muon import (
    PER_HEAD_PATTERNS,
    PerHeadMuon,
    _infer_head_count,
    _is_per_head_param,
    per_head_newton_schulz,
)
from rababa.training.optim import zeropower_via_newtonschulz5


def test_is_per_head_param_detects_attention_weights():
    p = nn.Parameter(torch.randn(768, 256))  # typical QKV shape
    assert _is_per_head_param("encoder.layers.0.qkv.weight", p)
    assert _is_per_head_param("encoder.layers.0.out_proj.weight", p)
    assert not _is_per_head_param("encoder.layers.0.w_gate.weight", p)
    assert not _is_per_head_param("encoder.embedding.weight", p)


def test_infer_head_count_for_common_shapes():
    # 8 heads × 32 head_dim × 256 dim → (256, 256) out_proj
    p = nn.Parameter(torch.randn(256, 256))
    heads = _infer_head_count("out_proj.weight", p)
    assert heads in (8, 4, 16)  # any valid factorization


def test_per_head_newton_schulz_handles_qkv():
    # Fused QKV: (3 * heads * head_dim, dim) = (3*8*32, 256) = (768, 256)
    G = torch.randn(768, 256)
    out = per_head_newton_schulz(G, "qkv.weight", heads=8, steps=5)
    assert out.shape == G.shape
    assert torch.isfinite(out).all()


def test_per_head_newton_schulz_handles_out_proj():
    # out_proj: (dim, dim) = (256, 256)
    G = torch.randn(256, 256)
    out = per_head_newton_schulz(G, "out_proj.weight", heads=8, steps=5)
    assert out.shape == G.shape
    assert torch.isfinite(out).all()


def test_per_head_newton_schulz_falls_back_when_heads_unknown():
    G = torch.randn(7, 11)  # primes, can't factor into heads
    out = per_head_newton_schulz(G, "qkv.weight", heads=None, steps=5)
    # Should fall back to whole-matrix NS.
    expected = zeropower_via_newtonschulz5(G, steps=5)
    assert torch.allclose(out, expected, atol=1e-5)


def test_per_head_muon_step_preserves_param_shape():
    """PerHeadMuon step should not change param shapes."""
    p = nn.Parameter(torch.randn(768, 256))
    opt = PerHeadMuon([p], lr=0.01, heads_hint=8)
    opt._param_names = {id(p): "encoder.layers.0.qkv.weight"}
    p.grad = torch.randn_like(p)
    original_shape = p.shape
    opt.step()
    assert p.shape == original_shape


def test_per_head_muon_skips_nan_grads():
    """PerHeadMuon should skip params with NaN grads (no weight corruption)."""
    p = nn.Parameter(torch.randn(64, 64))
    opt = PerHeadMuon([p], lr=0.01)
    opt._param_names = {id(p): "encoder.layers.0.qkv.weight"}
    p.grad = torch.full_like(p, float("nan"))
    original = p.clone()
    opt.step()
    assert torch.equal(p, original)  # unchanged

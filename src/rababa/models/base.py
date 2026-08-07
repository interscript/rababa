"""Diacritizer protocol — single interface for single- and multi-head models.

The training loop, ONNX exporter, and benchmark harness all consume
this protocol. Single-head models (Arabic) return a 1-element list
from `forward_heads` with a single head name; multi-head models
(Hebrew) return N.

`forward` (the ONNX-facing method) MAY return either a single Tensor
or a list[Tensor] — torch.onnx.export handles both. Callers that need
a uniform shape use `forward_heads`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Diacritizer(Protocol):
    """Common interface for diacritization models."""

    def forward_heads(self, src: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        """Return per-head logits. Always a list — len == len(head_names())."""
        ...

    def head_names(self) -> list[str]:
        """Output names in canonical order. Used as ONNX output_names and as
        keys for per-head DER in the benchmark."""
        ...


def build_model(cfg: dict) -> nn.Module:
    """Dispatch on cfg.model.arch. Returns a Diacritizer-conforming module."""
    from .modern import build_modern_student
    from .multi_head import build_multi_head_student
    from .student import build_student

    arch = cfg.get("model", {}).get("arch", "single")
    if arch == "modern":
        return build_modern_student(cfg)
    if arch == "multi_head":
        return build_multi_head_student(cfg)
    if arch in ("single", None):
        return build_student(cfg)
    raise ValueError(f"unknown model arch: {arch!r}")

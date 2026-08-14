"""TFLite export — PyTorch → .tflite via litert_torch (formerly ai-edge-torch).

Mirrors `export.py` but produces .tflite files runnable by LiteRT.js in
the browser. Same model architecture, same I/O contract — different
serialization format for a different runtime ecosystem.

Quantization: int8 (PT2E) is supported via `litert_torch.quantize.QuantConfig`.
For the spike we ship fp32 .tflite first; int8 is a follow-up pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models.base import build_model


class _TupleOutputWrapper(nn.Module):
    """Wrap a model that returns list[Tensor] to return tuple[Tensor, ...].

    litert_torch's exporter (and torch.export underneath) doesn't always
    handle Python list returns cleanly — tuple outputs trace better.
    Single-head models pass through unchanged.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor, lengths: torch.Tensor):
        out = self.model(src, lengths)
        if isinstance(out, list):
            return tuple(out)
        return out


def export_student_tflite(
    model_state_path: Path,
    cfg: dict[str, Any],
    out_path: Path,
    batch_size: int = 32,
    max_len: int = 200,
) -> None:
    """Export a trained student to TFLite (fp32). Same I/O as the ONNX export.

    Args:
        model_state_path: path to a `.pt` file containing `model.state_dict()`.
        cfg: model config dict (same as ONNX export).
        out_path: destination `.tflite` path.
        batch_size: fixed batch dimension.
        max_len: fixed seq dimension.
    """
    import litert_torch  # local import — heavy dep, only needed for this path

    model = build_model(cfg).eval()
    state = torch.load(model_state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    wrapped = _TupleOutputWrapper(model).eval()
    src = torch.randint(1, 50, (batch_size, max_len), dtype=torch.long)
    lengths = torch.full((batch_size,), max_len, dtype=torch.long)

    edge_model = litert_torch.convert(
        wrapped,
        sample_args=(src, lengths),
        # int64 inputs are downcast to int32 by default — our vocab IDs
        # fit comfortably in int32. Keep enable_x64=True to preserve
        # int64 if a future model needs it.
        enable_x64=True,
    )
    edge_model.export(str(out_path))


def export_student_tflite_int8(
    model_state_path: Path,
    cfg: dict[str, Any],
    out_path: Path,
    batch_size: int = 32,
    max_len: int = 200,
) -> None:
    """Export with PT2E int8 quantization. Currently a stub — the PT2E
    quantizer requires a calibration dataset and recipe setup that is
    not yet wired. Falls back to fp32 export with a warning."""
    import warnings

    warnings.warn(
        "TFLite int8 quantization via PT2E is not yet wired; "
        "exporting fp32 instead. See TODO.modernize/06a-litert-spike.md.",
        stacklevel=2,
    )
    export_student_tflite(model_state_path, cfg, out_path, batch_size, max_len)

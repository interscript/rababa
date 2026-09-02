"""ONNX export + int8 quantization.

Produces a fixed-shape ONNX file that the TS / Ruby runtimes can load.
The shape `[batch_size, max_len]` is fixed at export time — no dynamic
axes. This matches the trained rababa model's expectations and lets
the runtime replicate-batch-pad the input.

Works for both single-head (Arabic: 1 output "output") and multi-head
(Hebrew: 3 outputs niqqud/dagesh/sin) models via the `Diacritizer`
protocol's `head_names()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models.base import build_model


def export_student_onnx(
    model_state_path: Path,
    cfg: dict[str, Any],
    out_path: Path,
    batch_size: int = 32,
    max_len: int = 200,
) -> None:
    """Export a trained student to ONNX with fixed shape.

    Args:
        model_state_path: path to a `.pt` file containing `model.state_dict()`.
        cfg: model config dict (used to build the same architecture).
        out_path: destination `.onnx` path.
        batch_size: fixed batch dimension (default 32, matches runtime).
        max_len: fixed seq dimension (default 200, matches max_len trained).
    """
    model = build_model(cfg).eval()
    state = torch.load(model_state_path, map_location="cpu", weights_only=True)
    # Tolerate both raw state_dict and wrapped {"model": ...} formats.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)

    head_names = model.head_names()
    src = torch.randint(1, 50, (batch_size, max_len), dtype=torch.long)
    lengths = torch.full((batch_size,), max_len, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (src, lengths),
            str(out_path),
            opset_version=17,
            input_names=["src", "lengths"],
            output_names=head_names,
            dynamic_axes={},  # fully fixed shape
        )


# Alias — the export is task-agnostic now.
export_diacritizer_onnx = export_student_onnx


def quantize_dynamic_int8(in_path: Path, out_path: Path) -> None:
    """Apply dynamic int8 quantization to ONNX weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(in_path),
        str(out_path),
        weight_type=QuantType.QInt8,
    )


def quantize_static_int8(
    in_path: Path,
    out_path: Path,
    calibration_loader: Any,
    cfg: dict[str, Any],
    batch_size: int = 32,
    max_len: int = 200,
) -> None:
    """Apply static int8 quantization using a calibration DataLoader.

    Static quantization is preferred over dynamic for transformer models
    because it can also quantize activations. Requires a calibration set
    of ~1-5K representative inputs.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter(calibration_loader)
            self._enum = iter(self._generate())

        def _generate(self) -> list[dict[str, object]]:
            out: list[dict[str, object]] = []
            for batch in calibration_loader:
                import torch
                src_tensor = batch.src if hasattr(batch, "src") else batch[0]
                len_tensor = batch.lengths if hasattr(batch, "lengths") else batch[1]
                out.append({
                    "src": src_tensor[:batch_size, :max_len].numpy().astype("int64"),
                    "lengths": len_tensor[:batch_size].numpy().astype("int64"),
                })
                if len(out) >= 500:
                    break
            return out

        def get_next(self) -> dict[str, object] | None:
            try:
                return next(self._enum)
            except StopIteration:
                return None

    reader = _Reader()
    quantize_static(
        str(in_path),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )


def quantize_dynamic_int8(in_path: Path, out_path: Path) -> None:
    """Apply dynamic int8 quantization to ONNX weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(in_path),
        str(out_path),
        weight_type=QuantType.QInt8,
    )


def quantize_static_int8(
    in_path: Path,
    out_path: Path,
    calibration_loader: Any,
    cfg: dict[str, Any],
    batch_size: int = 32,
    max_len: int = 200,
) -> None:
    """Apply static int8 quantization using a calibration DataLoader.

    Static quantization is preferred over dynamic for transformer models
    because it can also quantize activations. Requires a calibration set
    of ~1-5K representative inputs.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter(calibration_loader)
            self._enum = iter(self._generate())

        def _generate(self) -> list[dict[str, object]]:
            out: list[dict[str, object]] = []
            for batch in calibration_loader:
                import torch
                src_tensor = batch.src if hasattr(batch, "src") else batch[0]
                len_tensor = batch.lengths if hasattr(batch, "lengths") else batch[1]
                out.append({
                    "src": src_tensor[:batch_size, :max_len].numpy().astype("int64"),
                    "lengths": len_tensor[:batch_size].numpy().astype("int64"),
                })
                if len(out) >= 500:
                    break
            return out

        def get_next(self) -> dict[str, object] | None:
            try:
                return next(self._enum)
            except StopIteration:
                return None

    reader = _Reader()
    quantize_static(
        str(in_path),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

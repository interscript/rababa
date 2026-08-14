"""TFLite export smoke tests — verify single-head and multi-head models
export to .tflite via litert_torch.

Skipped if litert_torch is not installed.

Run: pytest tests/test_tflite.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

litert_torch = pytest.importorskip("litert_torch")

from rababa.export_tflite import (  # noqa: E402
    _TupleOutputWrapper,
    export_student_tflite,
)
from rababa.models import build_model  # noqa: E402


def _save_state(model: torch.nn.Module, tmpdir: Path) -> Path:
    ckpt = tmpdir / "tiny.pt"
    torch.save(model.state_dict(), ckpt)
    return ckpt


# ---- Single-head (Arabic) -------------------------------------------

def test_tflite_export_single_head():
    cfg = {"model": {"arch": "single", "dim": 32, "layers": 1, "heads": 2,
                     "ff_dim": 64, "max_len": 16}}
    model = build_model(cfg)
    with TemporaryDirectory() as tmp:
        ckpt = _save_state(model, Path(tmp))
        out = Path(tmp) / "single.tflite"
        export_student_tflite(ckpt, cfg, out, batch_size=2, max_len=16)
        assert out.is_file()
        # Tiny model → small file, but should be > 1KB (flatbuffer overhead).
        assert out.stat().st_size > 1000


def test_tflite_export_multi_head():
    """Multi-head models return list[Tensor] — wrapper converts to tuple."""
    cfg = {"model": {"arch": "multi_head", "dim": 32, "layers": 1, "heads": 2,
                     "ff_dim": 64, "max_len": 16, "head_sizes": [16, 3, 4]}}
    model = build_model(cfg)
    with TemporaryDirectory() as tmp:
        ckpt = _save_state(model, Path(tmp))
        out = Path(tmp) / "multi.tflite"
        export_student_tflite(ckpt, cfg, out, batch_size=2, max_len=16)
        assert out.is_file()
        assert out.stat().st_size > 1000


# ---- Tuple wrapper ---------------------------------------------------

def test_tuple_output_wrapper_passthrough_single():
    """Single-output models pass through unchanged (no tuple wrapping)."""
    cfg = {"model": {"arch": "single", "dim": 32, "layers": 1, "heads": 2,
                     "ff_dim": 64, "max_len": 16}}
    model = build_model(cfg).eval()
    wrapped = _TupleOutputWrapper(model).eval()
    src = torch.randint(1, 20, (2, 8), dtype=torch.long)
    lengths = torch.full((2,), 8, dtype=torch.long)
    out = wrapped(src, lengths)
    # Single-head returns Tensor (not tuple)
    assert isinstance(out, torch.Tensor)


def test_tuple_output_wrapper_wraps_multi():
    """Multi-output models return tuple instead of list."""
    cfg = {"model": {"arch": "multi_head", "dim": 32, "layers": 1, "heads": 2,
                     "ff_dim": 64, "max_len": 16, "head_sizes": [16, 3, 4]}}
    model = build_model(cfg).eval()
    wrapped = _TupleOutputWrapper(model).eval()
    src = torch.randint(1, 20, (2, 8), dtype=torch.long)
    lengths = torch.full((2,), 8, dtype=torch.long)
    out = wrapped(src, lengths)
    assert isinstance(out, tuple)
    assert len(out) == 3

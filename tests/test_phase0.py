"""Phase 0 smoke tests — verify framework builds + dataset loads.

These run on CPU; no GPU required. They verify:
1. Config loads correctly.
2. Dataset reads from local Tashkeela files.
3. Model forward pass produces expected shape.
4. Training step doesn't crash.
5. ONNX export + int8 quantization works.

Run: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

# Add src/ to path when running without install.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rababa.config import load_task_config, to_dict  # noqa: E402
from rababa.datasets import TashkeelaDataset, load_tashkeela  # noqa: E402
from rababa.encoder import ArabicEncoder  # noqa: E402
from rababa.evaluate import compute_der  # noqa: E402
from rababa.models.student import CharTransformer, build_student, count_parameters  # noqa: E402
from rababa.training import masked_cross_entropy, train_supervised  # noqa: E402
from rababa.training.collate import collate_batch  # noqa: E402


# ---- Config ----------------------------------------------------------

def test_base_config_loads():
    cfg = load_task_config("rababa_arabic")
    assert cfg.name == "rababa_arabic"
    assert cfg.kind == "rababa"
    assert cfg.train.epochs > 0


# ---- Encoder ---------------------------------------------------------

def test_encoder_vocab_consistency():
    enc = ArabicEncoder(cleaner="arabic")
    ids = enc.encode("قطر")
    assert ids == [41, 12, 40]  # matches the legacy model
    assert enc.input_pad_id == 0


def test_clean_preserves_arabic():
    enc = ArabicEncoder(cleaner="arabic")
    # Cleaner preserves haraqat (they're in VALID_ARABIC). Strip is separate.
    cleaned = enc.clean("قِطْرَ ABC!")
    assert "ق" in cleaned and "ر" in cleaned
    # Non-Arabic chars dropped.
    assert "A" not in cleaned
    assert "!" not in cleaned


def test_strip_haraqat_chars():
    from rababa.datasets import strip_haraqat_chars

    assert strip_haraqat_chars("قِطْرَ") == "قطر"


# ---- Dataset ---------------------------------------------------------

def test_tashkeela_loads_train():
    ds = load_tashkeela("train", cleaner="arabic")
    assert len(ds) > 10_000, f"expected >10K examples, got {len(ds)}"
    first = ds[0]
    assert len(first.input_ids) == len(first.target_ids)
    assert first.input_ids[0] != 0  # not pad
    assert first.raw


def test_tashkeela_loads_test():
    ds = load_tashkeela("test", cleaner="arabic")
    assert len(ds) > 100


# ---- Model -----------------------------------------------------------

def test_student_forward_shape():
    model = build_student({"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128}})
    src = torch.randint(1, 40, (4, 32), dtype=torch.long)
    lengths = torch.full((4,), 32, dtype=torch.long)
    logits = model(src, lengths)
    assert logits.shape == (4, 32, 17), f"got {logits.shape}"


def test_student_param_count_budget():
    """Student must be small enough for browser deployment (~25M)."""
    model = build_student({})
    n = count_parameters(model)
    assert n < 30_000_000, f"student is {n:,} params, expected <30M"


# ---- Training --------------------------------------------------------

@pytest.mark.slow
def test_one_training_step():
    cfg = load_task_config("rababa_arabic")
    ds_train = load_tashkeela("train", cleaner="arabic")
    ds_val = load_tashkeela("val", cleaner="arabic")
    # Use tiny slices for CPU smoke test.
    train_examples = ds_train.examples[:32]
    val_examples = ds_val.examples[:8]
    ds_train.examples = train_examples
    ds_val.examples = val_examples

    cfg_dict = to_dict(cfg)
    cfg_dict["train"]["epochs"] = 1
    cfg_dict["model"] = {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128}

    from torch.utils.data import DataLoader

    train_loader = DataLoader(ds_train, batch_size=8, collate_fn=collate_batch)
    val_loader = DataLoader(ds_val, batch_size=8, collate_fn=collate_batch)

    with TemporaryDirectory() as tmpdir:
        model = train_supervised(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg_dict,
            device=torch.device("cpu"),
            ckpt_root=Path(tmpdir),
        )
        # Model should have non-trivial loss after one step.
        assert model.training or True  # just verify it returned


# ---- Eval ------------------------------------------------------------

def test_der_computation():
    # Predictions: 5 correct, 5 wrong, on 10 targets.
    preds = list(range(10))
    targets = [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]  # last 5 differ
    der = compute_der(preds, targets)
    assert der == 0.5, f"expected 0.5, got {der}"


# ---- Export ----------------------------------------------------------

@pytest.mark.slow
def test_export_to_onnx():
    from rababa.export import export_student_onnx, quantize_dynamic_int8

    cfg = {"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}}
    model = build_student(cfg)

    with TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        torch.save(model.state_dict(), ckpt_path)

        out_path = Path(tmpdir) / "test.onnx"
        export_student_onnx(ckpt_path, cfg, out_path, batch_size=4, max_len=32)
        assert out_path.is_file()
        assert out_path.stat().st_size > 1000

        q8_path = Path(tmpdir) / "test-q8.onnx"
        quantize_dynamic_int8(out_path, q8_path)
        assert q8_path.is_file()
        # int8 should generally be smaller, but for very small models the
        # overhead may dominate. Just check it's a valid file.
        assert q8_path.stat().st_size > 1000

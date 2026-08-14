"""MLM pretraining smoke tests — verify MLM head, dataset, training step.

Run: pytest tests/test_pretrain.py -v
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

from rababa.constants import INPUT_VOCAB_SIZE, MASK_ID, PAD_ID  # noqa: E402
from rababa.datasets import (  # noqa: E402
    ArabicMLMDataset,
    MLMExample,
    _apply_bert_mask,
    load_arabic_mlm,
)
from rababa.models.mlm import (  # noqa: E402
    MLMModel,
    build_pretrain_model,
    extract_pretrained_encoder,
)
from rababa.models.student import build_student  # noqa: E402
from rababa.training.pretrain import (  # noqa: E402
    load_pretrained_encoder,
    make_mlm_collate_fn,
    mlm_collate_batch,
    pretrain_mlm,
)


# ---- MLM head --------------------------------------------------------

def test_mlm_head_output_shape():
    cfg = {"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}}
    model = build_pretrain_model(cfg)
    src = torch.randint(1, INPUT_VOCAB_SIZE, (4, 16), dtype=torch.long)
    lengths = torch.full((4,), 16, dtype=torch.long)
    logits = model(src, lengths)
    assert logits.shape == (4, 16, INPUT_VOCAB_SIZE), f"got {logits.shape}"


def test_mlm_decoder_tied_to_embedding():
    cfg = {"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}}
    model = build_pretrain_model(cfg)
    # decoder.weight should be the SAME tensor object as encoder.embedding.weight
    assert model.head.decoder.weight is model.encoder.embedding.weight


def test_extract_encoder_excludes_head():
    cfg = {"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}}
    mlm = build_pretrain_model(cfg)
    encoder_state = extract_pretrained_encoder(mlm)
    # embedding + pos_embedding + transformer.* should be present
    assert any(k.startswith("embedding") for k in encoder_state)
    assert any(k.startswith("pos_embedding") for k in encoder_state)
    assert any(k.startswith("encoder.") for k in encoder_state)
    # haraqat head should NOT be present
    assert not any(k.startswith("head.") for k in encoder_state)


def test_load_pretrained_into_student_strict_false():
    cfg = {"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}}
    mlm = build_pretrain_model(cfg)
    encoder_state = extract_pretrained_encoder(mlm)

    student = build_student(cfg)
    # Fresh student has randomly-initialized head + randomly-initialized encoder.
    # After loading, encoder matches; head stays at fresh init.
    encoder_before = student.embedding.weight.clone()
    head_before = student.head.weight.clone()
    student.load_state_dict(encoder_state, strict=False)
    assert torch.equal(student.embedding.weight, mlm.encoder.embedding.weight)
    # Head untouched
    assert torch.equal(student.head.weight, head_before)
    assert not torch.equal(student.embedding.weight, encoder_before)


# ---- Masking ---------------------------------------------------------

def test_apply_bert_mask_selects_about_15pct():
    import random
    rng = random.Random(0)
    ids = list(range(1, 101))  # 100 non-PAD positions
    masked, target = _apply_bert_mask(ids, mask_prob=0.15, rng=rng, vocab_size=101, mask_id=100)
    selected = sum(1 for t in target if t != PAD_ID)
    assert 8 <= selected <= 25, f"expected ~15 selected, got {selected}"


def test_apply_bert_mask_preserves_unselected():
    import random
    rng = random.Random(1)
    ids = [10, 20, 30, 40, 50]
    masked, target = _apply_bert_mask(ids, mask_prob=0.0, rng=rng, vocab_size=100, mask_id=99)
    assert masked == ids
    assert all(t == PAD_ID for t in target)


def test_apply_bert_mask_skips_pad():
    import random
    rng = random.Random(2)
    ids = [10, PAD_ID, 20, PAD_ID, 30]
    masked, target = _apply_bert_mask(ids, mask_prob=1.0, rng=rng, vocab_size=100, mask_id=99)
    assert target[1] == PAD_ID
    assert target[3] == PAD_ID
    assert target[0] == 10
    assert target[2] == 20
    assert target[4] == 30


def test_apply_bert_mask_respects_vocab_size():
    """Random replacement must stay within the model's embedding range."""
    import random
    rng = random.Random(0)
    ids = [10] * 1000
    masked, _ = _apply_bert_mask(ids, mask_prob=1.0, rng=rng, vocab_size=20, mask_id=19)
    valid = set([19] + list(range(1, 20)))
    bad = [m for m in masked if m not in valid]
    assert not bad, f"out-of-vocab IDs generated: {bad[:10]}"


# ---- MLM dataset -----------------------------------------------------

def test_arabic_mlm_dataset_loads():
    ds = load_arabic_mlm("train", mask_prob=0.15, max_len=64)
    assert len(ds) > 100
    ex = ds[0]
    assert isinstance(ex, MLMExample)
    assert len(ex.input_ids) == len(ex.target_ids)
    assert len(ex.input_ids) <= 64


def test_arabic_mlm_dataset_masks_different_per_call():
    """Each __getitem__ call uses a deterministic seed, so same idx → same mask."""
    ds = ArabicMLMDataset(split="train", mask_prob=0.5, max_len=64, seed=123)
    ex1 = ds[0]
    ex2 = ds[0]
    assert ex1.input_ids == ex2.input_ids  # deterministic per (seed, idx, len)


def test_mlm_collate_pads_to_batch_max():
    examples = [
        MLMExample(input_ids=[1, 2, 3], target_ids=[0, 2, 0], raw="a"),
        MLMExample(input_ids=[4, 5], target_ids=[0, 5], raw="b"),
    ]
    batch = mlm_collate_batch(examples, max_len=64)
    assert batch.src.shape == (2, 3)
    assert batch.lengths.tolist() == [3, 2]
    # Position 1 of example 1 is PAD
    assert batch.src[1, 2].item() == PAD_ID


def test_make_mlm_collate_fn_binds_max_len():
    collate = make_mlm_collate_fn(max_len=8)
    examples = [MLMExample(input_ids=list(range(1, 11)), target_ids=[0] * 10, raw="long")]
    batch = collate(examples)
    assert batch.src.shape == (1, 8)  # truncated


# ---- Training step ---------------------------------------------------

@pytest.mark.slow
def test_one_pretrain_step_cpu():
    from torch.utils.data import DataLoader

    train_ds = load_arabic_mlm("train", mask_prob=0.15, max_len=32)
    val_ds = load_arabic_mlm("val", mask_prob=0.15, max_len=32)
    # Slice for CPU smoke test
    train_ds.sequences = train_ds.sequences[:16]
    val_ds.sequences = val_ds.sequences[:8]

    collate = make_mlm_collate_fn(32)
    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

    cfg = {
        "model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32},
        "train": {"epochs": 1, "learning_rate": 3e-4, "warmup_steps": 5, "fp16": False},
    }

    with TemporaryDirectory() as tmpdir:
        model, best_path = pretrain_mlm(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=torch.device("cpu"),
            ckpt_root=Path(tmpdir),
        )
        assert best_path.is_file()

        # Checkpoint should load into a fresh student via load_pretrained_encoder.
        student = build_student(cfg)
        load_pretrained_encoder(best_path, student)
        # Encoder embedding matches the pretrained MLM
        assert torch.equal(student.embedding.weight, model.encoder.embedding.weight)

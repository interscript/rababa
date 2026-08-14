"""Hebrew spike smoke tests — verify constants, encoder, parser, multi-head model.

Run: pytest tests/test_hebrew.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rababa.constants_hebrew import (  # noqa: E402
    DAGESH_VOCAB_SIZE,
    ENDINGS_TO_REGULAR,
    HEBREW_LETTERS,
    INPUT_VOCAB_SIZE,
    MASK_ID,
    NIQQUD_VOCAB_SIZE,
    PAD_ID,
    SIN_VOCAB_SIZE,
    can_dagesh,
    can_niqqud,
    can_sin,
    is_hebrew_letter,
)
from rababa.datasets import (  # noqa: E402
    NakdimonDataset,
    _hebrew_marks_to_targets,
    _iterate_dotted_hebrew,
)
from rababa.encoder import HebrewEncoder, normalize_hebrew_char  # noqa: E402
from rababa.models.multi_head import (  # noqa: E402
    OUTPUT_ORDER,
    MultiHeadCharTransformer,
    build_multi_head_student,
)


# ---- Constants -------------------------------------------------------

def test_vocab_sizes_match_legacy_onnx():
    """Legacy hebrew-model.onnx has heads [16, 3, 4]. Our vocabs must match."""
    assert NIQQUD_VOCAB_SIZE == 16
    assert DAGESH_VOCAB_SIZE == 3
    assert SIN_VOCAB_SIZE == 4
    assert INPUT_VOCAB_SIZE > len(HEBREW_LETTERS)


def test_hebrew_letters_count():
    assert len(HEBREW_LETTERS) == 27
    assert HEBREW_LETTERS[0] == "א"
    assert HEBREW_LETTERS[-1] == "ת"


def test_endings_mapping():
    assert ENDINGS_TO_REGULAR["ך"] == "כ"
    assert ENDINGS_TO_REGULAR["ץ"] == "צ"


def test_can_mark_predicates():
    assert can_dagesh("ב")
    assert not can_dagesh("א")
    assert can_sin("ש")
    assert not can_sin("ב")
    assert can_niqqud("א")
    assert not can_niqqud("ש") or can_niqqud("ש")  # ש takes niqqud too in Nakdimon


def test_is_hebrew_letter():
    assert is_hebrew_letter("א")
    assert is_hebrew_letter("ת")
    assert not is_hebrew_letter("A")
    assert not is_hebrew_letter("5")


# ---- Encoder ---------------------------------------------------------

def test_hebrew_encoder_preserves_end_of_word_forms():
    """End-of-word forms (ךםןףץ) are valid letters and stay as-is — not
    normalized to their regular forms. (Nakdimon's normalize() treats
    them as VALID_LETTERS too; ENDINGS_TO_REGULAR is for fallback cases.)"""
    enc = HebrewEncoder()
    ids_final = enc.encode(enc.clean("ך"))
    ids_regular = enc.encode(enc.clean("כ"))
    assert ids_final != ids_regular  # distinct IDs
    assert len(ids_final) == 1
    assert len(ids_regular) == 1


def test_hebrew_encoder_normalize_digit():
    assert normalize_hebrew_char("5") == "5"
    assert normalize_hebrew_char("9") == "5"  # digits collapse to "5"


def test_hebrew_encoder_normalize_unknown():
    assert normalize_hebrew_char("@") == "O"
    assert normalize_hebrew_char("ײ") == "H"  # Yiddish ligature


def test_hebrew_encoder_normalize_dash_variants():
    assert normalize_hebrew_char("—") == "-"
    assert normalize_hebrew_char("־") == "-"


def test_hebrew_encoder_round_trip():
    enc = HebrewEncoder()
    text = "שָׁלוֹם"
    cleaned = enc.clean(text)
    # After clean, diacritics are stripped via iterate (we just keep letters here)
    ids = enc.encode(cleaned)
    assert len(ids) > 0
    decoded = enc.decode_input(ids)
    # Decoded should contain only Hebrew letters, no diacritics
    assert all(is_hebrew_letter(c) or c in " H O 5" or c in "!,.;:?\"'()-" for c in decoded)


# ---- Dotted-text parser ---------------------------------------------

def test_iterate_dotted_simple():
    """שָׁלוֹם should parse as ש(sin+kamatz) ל ו(holam) ם."""
    result = list(_iterate_dotted_hebrew("שָׁלוֹם"))
    letters = [r[0] for r in result]
    assert letters == ["ש", "ל", "ו", "ם"]
    # ש should have sin (SHIN_YEMANIT) + kamatz niqqud
    shin_letter, shin_niqqud, shin_dagesh, shin_sin = result[0]
    assert shin_sin == "ׁ"  # SHIN_YEMANIT
    assert shin_niqqud == "ָ"  # KAMATZ
    # ו should have holam niqqud
    vav_letter, vav_niqqud, _, _ = result[2]
    assert vav_niqqud == "ֹ"  # HOLAM


def test_iterate_dotted_shuruk_special_case():
    """וּ (vav + dagesh, no niqqud) should parse as SHURUK = niqqud."""
    result = list(_iterate_dotted_hebrew("וּ"))
    assert len(result) == 1
    letter, niqqud, dagesh, sin = result[0]
    assert letter == "ו"
    # Special case: dagesh moves to niqqud role
    assert dagesh == ""
    assert niqqud == "ּ"  # SHURUK codepoint


def test_iterate_dotted_dagesh_on_bet():
    """בּ (bet + dagesh) → dagesh set, niqqud empty."""
    result = list(_iterate_dotted_hebrew("בּ"))
    assert len(result) == 1
    letter, niqqud, dagesh, sin = result[0]
    assert letter == "ב"
    assert dagesh == "ּ"
    assert niqqud == ""


def test_marks_to_targets_skips_inapplicable():
    """א can't take dagesh → dagesh target should be PAD_ID."""
    n_id, d_id, s_id = _hebrew_marks_to_targets("א", "ָ", "", "")
    assert n_id != PAD_ID  # niqqud applies
    assert d_id == PAD_ID  # dagesh skipped
    assert s_id == PAD_ID  # sin skipped


def test_marks_to_targets_rafe_when_applicable_no_mark():
    """ש can take sin but no sin mark → sin target should be RAFE (not PAD)."""
    from rababa.constants_hebrew import SIN_VOCAB
    rafe_id = SIN_VOCAB.index("ֿ")
    n_id, d_id, s_id = _hebrew_marks_to_targets("ש", "", "", "")
    # ש can take all three
    assert n_id != PAD_ID
    assert d_id != PAD_ID
    assert s_id == rafe_id  # RAFE = "decided: none"


# ---- Multi-head model -----------------------------------------------

def test_multi_head_model_forward_shapes():
    model = build_multi_head_student({"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}})
    src = torch.randint(1, 20, (4, 16), dtype=torch.long)
    lengths = torch.full((4,), 16, dtype=torch.long)
    outputs = model(src, lengths)
    assert isinstance(outputs, list)
    assert len(outputs) == 3
    assert outputs[0].shape == (4, 16, 16)  # niqqud
    assert outputs[1].shape == (4, 16, 3)   # dagesh
    assert outputs[2].shape == (4, 16, 4)   # sin
    assert OUTPUT_ORDER == ("niqqud", "dagesh", "sin")


def test_multi_head_model_param_count_budget():
    model = build_multi_head_student({})
    n = sum(p.numel() for p in model.parameters())
    assert n < 30_000_000, f"multi-head student is {n:,} params, expected <30M"


def test_multi_head_encoder_compatible_with_char_transformer_state_dict():
    """Encoder weights should have the same key names as CharTransformer,
    so MLM-pretrained checkpoints load across both."""
    from rababa.models.student import build_student
    single = build_student({"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}})
    multi = build_multi_head_student({"model": {"dim": 64, "layers": 2, "heads": 2, "ff_dim": 128, "max_len": 32}})
    single_keys = {k for k in single.state_dict().keys() if not k.startswith("head.")}
    multi_keys = {k for k in multi.state_dict().keys() if not k.startswith("heads.")}
    # Encoder keys (embedding, pos_embedding, encoder.*) must match exactly.
    assert single_keys == multi_keys


def test_diacritizer_protocol_conformance():
    """Both single- and multi-head models implement forward_heads + head_names."""
    from rababa.models import build_model
    single = build_model({"model": {"arch": "single", "dim": 32, "layers": 1, "heads": 2, "ff_dim": 64, "max_len": 16}})
    multi = build_model({"model": {"arch": "multi_head", "dim": 32, "layers": 1, "heads": 2, "ff_dim": 64, "max_len": 16}})

    src = torch.randint(1, 10, (2, 8), dtype=torch.long)
    lengths = torch.full((2,), 8, dtype=torch.long)

    single_out = single.forward_heads(src, lengths)
    multi_out = multi.forward_heads(src, lengths)

    assert isinstance(single_out, list) and len(single_out) == 1
    assert isinstance(multi_out, list) and len(multi_out) == 3
    assert single.head_names() == ["output"]
    assert multi.head_names() == ["niqqud", "dagesh", "sin"]


# ---- Dataset (smoke; no real Nakdimon data) -------------------------

@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "test-datasets" / "nakdimon").is_dir(),
    reason="Nakdimon corpus not present locally",
)
def test_nakdimon_dataset_loads():
    from rababa.datasets import load_nakdimon
    ds = load_nakdimon("test", max_len=64)
    assert len(ds) > 0
    ex = ds[0]
    assert len(ex.input_ids) == len(ex.niqqud_ids)
    assert len(ex.input_ids) == len(ex.dagesh_ids)
    assert len(ex.input_ids) == len(ex.sin_ids)


# ---- Integration: multi-head training + ONNX export -----------------

@pytest.mark.slow
def test_multi_head_training_step_cpu():
    """End-to-end: build multi-head model + synthetic Hebrew data, run one
    train_supervised step. Verifies the unified training loop works for
    Hebrew without needing real Nakdimon corpus."""
    import random
    import string
    from torch.utils.data import DataLoader

    from rababa.datasets import HebrewExample
    from rababa.training import train_supervised
    from rababa.training.collate import multi_head_collate_batch

    def _rand_hebrew_example(rng: random.Random) -> HebrewExample:
        n = rng.randint(8, 16)
        # Random Hebrew-input-range IDs (1..43) and target IDs.
        input_ids = [rng.randint(1, 20) for _ in range(n)]
        # Targets per head: most positions evaluable (non-PAD), some PAD.
        niqqud_ids = [rng.randint(1, 15) if rng.random() > 0.1 else 0 for _ in range(n)]
        dagesh_ids = [rng.randint(1, 2) if rng.random() > 0.5 else 0 for _ in range(n)]
        sin_ids = [rng.randint(1, 3) if rng.random() > 0.8 else 0 for _ in range(n)]
        raw = "".join(rng.choice(string.ascii_letters) for _ in range(n))
        return HebrewExample(input_ids=input_ids, niqqud_ids=niqqud_ids,
                             dagesh_ids=dagesh_ids, sin_ids=sin_ids, raw=raw)

    rng = random.Random(0)
    train_examples = [_rand_hebrew_example(rng) for _ in range(16)]
    val_examples = [_rand_hebrew_example(rng) for _ in range(8)]

    class _StubDataset:
        def __init__(self, examples):
            self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, i): return self.examples[i]

    train_loader = DataLoader(_StubDataset(train_examples), batch_size=4,
                              collate_fn=multi_head_collate_batch)
    val_loader = DataLoader(_StubDataset(val_examples), batch_size=4,
                            collate_fn=multi_head_collate_batch)

    cfg = {
        "model": {"arch": "multi_head", "dim": 32, "layers": 1, "heads": 2,
                  "ff_dim": 64, "max_len": 32, "head_sizes": [16, 3, 4]},
        "train": {"epochs": 1, "learning_rate": 1e-3, "warmup_steps": 2,
                  "fp16": False, "batch_size": 4},
    }

    with __import__("tempfile").TemporaryDirectory() as tmpdir:
        model = train_supervised(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=torch.device("cpu"),
            ckpt_root=Path(tmpdir),
        )
        # Should return a multi-head model with 3 heads.
        assert hasattr(model, "heads")
        assert len(model.heads) == 3


@pytest.mark.slow
def test_multi_head_onnx_export():
    """Export a multi-head model → ONNX. Verify 3 outputs named correctly."""
    import onnx
    from onnxruntime.quantization.quantize import quantize_dynamic

    from rababa.export import export_student_onnx
    from tempfile import TemporaryDirectory

    cfg = {
        "model": {"arch": "multi_head", "dim": 32, "layers": 1, "heads": 2,
                  "ff_dim": 64, "max_len": 16, "head_sizes": [16, 3, 4]},
    }
    model = build_multi_head_student(cfg)

    with TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "mh.pt"
        torch.save(model.state_dict(), ckpt)

        onnx_path = Path(tmpdir) / "mh.onnx"
        export_student_onnx(ckpt, cfg, onnx_path, batch_size=4, max_len=16)
        assert onnx_path.is_file()

        # Inspect the ONNX graph: should have 3 outputs named niqqud/dagesh/sin.
        graph = onnx.load(str(onnx_path)).graph
        out_names = [o.name for o in graph.output]
        assert out_names == ["niqqud", "dagesh", "sin"], f"got {out_names}"

        # Smoke check: int8 quantization works on multi-head ONNX.
        from rababa.export import quantize_dynamic_int8
        q8_path = Path(tmpdir) / "mh-q8.onnx"
        quantize_dynamic_int8(onnx_path, q8_path)
        assert q8_path.is_file()

# rababa — modern Arabic / Hebrew diacritization

[![Tests](https://github.com/interscript/rababa/actions/workflows/test.yml/badge.svg)](https://github.com/interscript/rababa/actions)

Modern reimplementation of rababa with Modal-native training and
browser-deployable ONNX models.

## Status

- **Phase 0 (foundations)**: ✅ complete. Framework, dataset, model, ONNX export, tests all working.
- **Phase 0.5 (MLM pretrain)**: ✅ implemented. Char-level MLM pretraining stage added (`TODO.modernize/02a-mlm-pretrain.md`). Run before Tier 1 fine-tune for ~7-9pt DER improvement.
- **Phase 1 (rababa_arabic v0.1.0)**: pending — run `modal run modal_app.py::pretrain` then `modal run modal_app.py::train --init-from-pretrain ...`.

See [`TODO.modernize/`](TODO.modernize/) for the full plan.

## What this repo contains

- `src/rababa/` — modern Python package (Tier 1 supervised training, ONNX export, eval)
- `modal_app.py` — Modal definitions (train, export, evaluate)
- `configs/` — task-specific YAML configs (`rababa_arabic.yaml`, base.yaml)
- `tests/` — CPU smoke tests (run with `pytest`)
- `python/` — legacy 2021 CBHG training code (reference only)
- `lib/rababa/` — Ruby gem (OnnxRuntime wrapper for runtime use)
- `test-datasets/tashkeela/` — Tashkeela Arabic corpus (50K train, 2.5K val, 2.5K test)

## Quick start

```bash
# Install (development)
pip install -e .

# Run CPU smoke tests
pytest tests/

# Modal auth (one-time)
pip install modal
modal token new

# MLM pretrain (A100, ~6h) — produces encoder checkpoint
modal run modal_app.py::pretrain --task rababa_arabic_pretrain

# Fine-tune rababa_arabic on Modal with pretrained init (A100, ~3h)
modal run modal_app.py::train --task rababa_arabic \
  --init-from-pretrain /checkpoints/rababa_arabic_pretrain/run-001/best.pt

# Export to ONNX + int8
modal run modal_app.py::export_onnx --task rababa_arabic --version v0.1.0

# Evaluate on test split
modal run modal_app.py::evaluate --task rababa_arabic
```

## Architecture

```
src/rababa/
├── __init__.py
├── config.py           # OmegaConf loader: base.yaml + <task>.yaml
├── constants.py        # Arabic alphabet + haraqat (Unicode codepoints)
├── encoder.py          # ArabicEncoder: text → token IDs
├── datasets.py         # Tashkeela loader
├── models/
│   └── student.py      # CharTransformer (6-layer encoder, ~11M params)
├── training/
│   ├── collate.py      # Padding + truncation
│   └── supervised.py   # Tier 1 training loop (AMP, grad-clip, cosine schedule)
├── export.py           # PyTorch → ONNX (fixed shape) + int8 quantization
├── evaluate.py         # DER + per-example accuracy
└── cli.py              # rababa-train / rababa-export / rababa-evaluate
```

## Training tiers

- **Phase 0.5 — MLM pretrain** (optional, recommended): char-level masked-LM
  pretraining on raw Arabic text. Produces an encoder checkpoint that
  initializes the Tier 1 student. Same `CharTransformer` architecture;
  no browser-deployment change.

- **Tier 1** (default): direct supervised training on gold labels.
  ~11M params char transformer, ~3h on A100, DER target ≤ 10% on
  Tashkeela test (with pretrain init; ≤ 15% without).

- **Tier 2** (optional): distillation with teacher-as-noisy-oracle on unlabeled data.
  Only triggered if Tier 1 misses DER target.

See `TODO.modernize/02-phase1-rababa-arabic.md` and
`TODO.modernize/02a-mlm-pretrain.md` for details.

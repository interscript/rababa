# Phase 1 — rababa Arabic (Tier 1: direct supervised)

## Goal
Cut `rababa_arabic-v0.1.0`. Direct supervised training of a 6-layer char
transformer on Tashkeela++ gold labels. No teacher in Tier 1.

## Why no teacher
A teacher trained on Tashkeela++ inherits Tashkeela++'s biases. It
doesn't know "the rules" beyond what's in the gold data. Distillation
from such a teacher transfers those biases to the student without
adding new information — unless the teacher is used as a noisy oracle
on UNLABELED data (Tier 2).

Tier 1 is direct supervised training: student learns from gold labels
directly. If Tier 1 doesn't hit DER targets, Tier 2 adds teacher-labeled
unlabeled data as augmentation.

## Tasks

### 1.1 Tier 1 student training
- Arch: 6-layer char transformer, 384 dim, 6 heads (~25 M params).
- Data: Tashkeela++ train split (~50 K pairs, deduped).
- Compute: 1× A100 40 GB, 5 epochs, ~3 h.
- Eval: DER / PER on Tashkeela++ test split (≥ 5 K pairs).
- Acceptance for v0.1.0: **DER ≤ 15%** on test (research baseline; v0.5.0 target ≤ 10%).

```python
# rababa/src/rababa/training/supervised.py
def main(task: str, epochs: int, fp16: bool, data_root: str, ckpt_root: str):
    cfg = load_task_config(task)
    model = build_student(cfg.model)
    train_loader, val_loader = build_dataloaders(task, data_root)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, total_steps=epochs * len(train_loader))

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler)
        metrics = evaluate(model, val_loader)
        save_checkpoint(model, ckpt_root, epoch, metrics)
        log_to_wandb(metrics)
```

### 1.2 ONNX export + int8 quantization
- Shape: `[batch_size=32, max_len=200]` (matches existing runtime).
- Quantization: int8 with calibration set (5 K random Tashkeela samples).
- Sizes: fp32 ~ 100 MB, fp16 ~ 50 MB, int8 ≤ 25 MB.
- Export: `torch.onnx.export` with `dynamic_axes={}` (fully fixed shape).
- Verify: 100-example parity test vs PyTorch (TS runner).

```python
# rababa/src/rababa/export.py
import torch
from torch.onnx import export
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_student(model, onnx_path: str, vocab_path: str):
    export(model, ("src", "lengths"),
           onnx_path, opset_version=17,
           input_names=["src","lengths"], output_names=["output"],
           dynamic_axes={})  # fully fixed shape
    quantize_dynamic(onnx_path, onnx_path.replace(".onnx", "-q8.onnx"),
                     weight_type=QuantType.QInt8)
```

### 1.3 Manifest + GitHub release
- Cut `rababa_arabic-v0.1.0` tag.
- Upload: `models/rababa_arabic-v0.1.0-q8.onnx`, `-vocab.json`, SHA256SUMS.
- Update `ml-models/npm/models/manifest.json`: version `0.1.0`, status `research`.
- jsDelivr auto-mirrors via `https://cdn.jsdelivr.net/gh/interscript/rababa@rababa_arabic-v0.1.0/`.

### 1.4 TS runtime update
```typescript
// src/stdlib/ml.ts — bump default URL
const DEFAULT_RABABA_CONFIGS = Object.freeze({
  "v0.1": Object.freeze({
    model: "https://cdn.jsdelivr.net/gh/interscript/rababa@rababa_arabic-v0.1.0/models/rababa_arabic-v0.1.0-q8.onnx",
    config: { max_len: 200, batch_size: 32 },
  }),
})
```
- Old `model-200.onnx` (secryst v0.1) kept as fallback.
- Test: existing end-to-end test passes against new model.

### 1.5 Ruby gem update
- Bump `rababa` to v0.3.0.
- Use new model path via `Interscript.rababa_configs["v0.1"]`.
- Specs pass against new model.

### 1.6 (Optional, deferred) Tier 2 distillation
Only triggered if Tier 1 v0.1.0 DER > 15% on test:
- Train a teacher (larger char transformer, same architecture family, ~100 M params).
- Use teacher to label ~500 K UNLABELED Arabic text (Common Crawl Arabic).
- Filter: keep only teacher predictions where teacher confidence > 0.95.
- Train student on `gold_data ∪ filtered_teacher_labeled_data`.
- Acceptance for Tier 2: student DER improves by > 2 absolute points on gold test.

## Acceptance (Tier 1 v0.1.0)

- [ ] DER ≤ 15% student on Tashkeela++ test split
- [ ] int8 ONNX ≤ 25 MB
- [ ] TS end-to-end parity: 100% on rababa test vectors (1 test vector)
- [ ] Ruby spec suite passes against new model
- [ ] Rollback path: bumping `version` back to `0.0.0` in manifest reverts to 2021 model

## Path to v0.5.0 (research-quality)

After v0.1.0 ships, the v0.5.0 sprint will:
1. Run Tier 2 distillation (if not done in v0.1.0).
2. Try larger student (12 layers, 768 dim, ~80 M params) — may not fit browser budget.
3. Data augmentation: character substitution, code-mixing for robustness.
4. Hyperparameter sweep: learning rate, warmup steps, dropout.

## Path to v1.0.0 (stable)

- DER ≤ 10% on Tashkeela++ test.
- 1 month of v0.5.0 deployment in production with no regressions.
- All website rababa tests pass.

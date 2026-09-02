# 43 — Data Loader Parallelism (num_workers bump)

## Problem

Current DataLoaders use `num_workers=2` (rababa) / `0` (secryst). For
Arabic Pro with 2.1M lines, this is a perf bottleneck — the GPU starves
while waiting for batches.

## Fix

Bump default to `num_workers=8` on Linux/Modal (CPU count typically 8+).
Add `cfg.train.num_workers` config knob.

Also enable `persistent_workers=True` and `pin_memory=True` for additional
speedup.

```python
train_loader = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=cfg_train.get("num_workers", 8),
    persistent_workers=True,
    pin_memory=True,
    collate_fn=collate,
)
```

## Files

- `src/rababa/tasks.py:build_supervised_loaders` + `build_mlm_loaders` — bump.
- `src/secryst/tasks.py:build_supervised_loaders` — bump.
- `configs/*.yaml` — add `train.num_workers: 8` (optional).

## Acceptance

- Arabic Pro pretrain epoch time decreases ~30-50% on Modal A100.
- No regressions in Hebrew (small dataset, may not help much).

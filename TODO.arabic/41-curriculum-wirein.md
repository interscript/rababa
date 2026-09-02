# 41 — Curriculum Learning Wire-In

## Problem

`CurriculumSampler` exists (`src/rababa/training/curriculum.py`) but isn't
wired into the supervised training loop. Activation requires manually
patching the DataLoader. Should be a config flag.

## Fix

Add `cfg.train.curriculum.enabled: true` flag. When enabled, the supervised
training loop wraps its DataLoader with `CurriculumSampler` and uses the
configured schedule (linear, sqrt) and difficulty signal.

```python
if cfg_train.get("curriculum", {}).get("enabled", False):
    from .curriculum import CurriculumSampler
    sampler = CurriculumSampler(
        dataset=train_loader.dataset,
        difficulty_fn=_difficulty_fn(cfg),
        schedule=cfg_train.curriculum.schedule,
        total_epochs=epochs,
    )
    train_loader = DataLoader(
        train_loader.dataset,
        sampler=sampler,
        batch_size=cfg_train.batch_size,
        ...
    )
```

## Files

- `src/rababa/training/supervised.py:train_supervised` — wrap loader if enabled.
- `configs/rababa_arabic_pro.yaml` — add `curriculum: {enabled: true, ...}`.
- `tests/training/test_curriculum_wirein.py` (NEW) — spec: flag toggles sampler.

## Acceptance

- `curriculum.enabled: false` (default) → no behavior change.
- `curriculum.enabled: true` → CurriculumSampler wraps the loader.
- Difficulty signal: iltiqaa_violation + word_boundary (Arabic) — reuse
  `compute_arabic_features` from `features/arabic.py`.

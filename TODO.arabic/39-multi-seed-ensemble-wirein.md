# 39 — Multi-Seed Ensemble + Distillation Wire-In

## Problem

`multi_seed.py` and `distill.py` exist but aren't wired into the SOTA
pipeline. Currently they're standalone scripts that require manual
invocation. For Hebrew/Arabic v1.0+, we want a single pipeline that
automatically:

1. Trains 3 students with different seeds.
2. Ensembles their outputs (averaging or voting).
3. Distills the ensemble into a single student via KL on temperature-
   scaled softmax (Hinton et al. 2015).

## Architecture

```
run_sota_pipeline (existing)
  ↓ stage: train  (1 model)
  ↓ stage: export
  ↓ stage: evaluate
  ↓ stage: multi_seed (NEW)
      ├─ train seed=1 → student_1
      ├─ train seed=2 → student_2
      └─ train seed=3 → student_3
  ↓ stage: ensemble_distill (NEW)
      └─ single student trained on KL(student || avg(teachers))
  ↓ stage: export_distilled (NEW)
  ↓ stage: evaluate_distilled (NEW)
```

The `multi_seed` stage runs N training jobs in parallel using
`modal.Function.map()`. Each job is identical to the normal `train` except
for the seed and the run-dir suffix.

## Files

- `src/rababa/training/multi_seed.py` — already exists; add `run_multi_seed`.
- `src/rababa/training/distill.py` — already exists; add `distill_from_ensemble`.
- `modal_app.py:run_sota_pipeline` — add new stages.
- `configs/rababa_*_ensemble.yaml` (NEW) — ensemble config (n_seeds, alpha).
- `tests/training/test_ensemble_wirein.py` (NEW) — specs.

## Config flag

```yaml
ensemble:
  enabled: true
  n_seeds: 3
  alpha: 0.5          # weight on distillation loss (0 = pure CE, 1 = pure KL)
  temperature: 4.0    # softmax temperature for KL
```

## Acceptance

- Pipeline runs multi_seed stage when `ensemble.enabled: true`.
- N seed checkpoints produced at `/checkpoints/{task}/seed-{i}/`.
- Distilled checkpoint at `/checkpoints/{task}/run-002/best.pt`.
- Distilled model has DER ≤ single-seed model (typically 5-10% improvement).
- All specs pass.

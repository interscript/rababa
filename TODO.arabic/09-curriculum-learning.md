# 09 — Curriculum learning

## Why
Standard training shuffles uniformly. Rare-haraqat examples
(Shaddah+Dammatan, Shaddah+Kasratan) get a tiny fraction of gradient
signal in early epochs. Curriculum learning sorts examples by
haraqat-density (rare combinations last), so the model first learns
the common case solidly, then refines on rare cases.

Cheap; usually good for ~2-3% DER.

## Tasks

### 9.1 Difficulty scorer (`src/rababa/training/curriculum.py`)
- `score_difficulty(example: Example) -> float`: returns 0 (easy) to
  1 (hard) based on rare-haraqat frequency.
- Backed by a precomputed rare-haraqat lookup table.

### 9.2 Curriculum sampler
- New `torch.utils.data.Sampler` that yields examples in difficulty
  buckets.
- `CurriculumSampler(dataset, n_buckets=5, schedule=linear)`.
- "Linear" means at epoch 0 we sample from bucket 0 only; by final
  epoch we sample uniformly.

### 9.3 Wire into DataLoader
- `cfg.train.curriculum: "none" | "linear" | "sqrt"` (default none).
- When non-none, replace `shuffle=True` with `CurriculumSampler`.

## Acceptance
- [ ] Curriculum sampler covers all examples over the training run.
- [ ] Per-haraqat-class DER improves on rare classes without
      regressing common classes.

## Files
- `src/rababa/training/curriculum.py` (new)
- `src/rababa/training/supervised.py` (use sampler when configured)
- `tests/training/test_curriculum.py` (new)

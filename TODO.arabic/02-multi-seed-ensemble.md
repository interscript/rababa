# 02 — Multi-seed ensemble + distillation

## Why
Single-model training has run-to-run variance of ~1% DER from random
init + data shuffle noise. Averaging 3 seeds via distillation typically
drops DER 10-15% with no architecture change.

## Architecture

```
            ┌─ run-001 (seed 42)   ──┐
            │                         │
arabic-pro ├─ run-002 (seed 1337) ──┼─→ distill → run-ensemble → ship
            │                         │
            └─ run-003 (seed 2026)  ──┘
```

Distillation = soft-label KL loss against the ensemble's averaged
probabilities, plus the standard CE loss against gold labels.

## Tasks

### 2.1 Multi-seed launcher (`scripts/train_seeds.py`)
- Accept `--task`, `--seeds 42,1337,2026`, run them in parallel via
  Modal `.starmap()`.
- Each seed writes to `/checkpoints/{task}/run-{seed}/`.
- The launcher is a thin wrapper around the existing `train` function.

### 2.2 Distillation training loop (`src/rababa/training/distill.py`)
- New module: loads N teacher checkpoints, averages logits per batch,
  trains a fresh student with `(1-α)·CE(student, gold) + α·KL(student, teacher_avg)`.
- α schedule: linear from 0.5 → 0 across training (start with teacher
  guidance, end on gold).
- Reuses `train_supervised` infrastructure: same optimizer, scheduler,
  checkpoint resume, log_fn.

### 2.3 Wire into Modal (`modal_app.py::distill`)
- New `@app.function` that:
  1. Loads N teacher checkpoints from `/checkpoints/{task}/run-{seed}*/best.pt`.
  2. Builds the student with the same arch.
  3. Calls `distill_into_student(teachers, student, ...)`.
  4. Saves to `/checkpoints/{task}/run-distill/best.pt`.

### 2.4 Add `distill` stage to `run_sota_pipeline`
- New stage between `train` and `export`.
- Reads `--n-seeds` to know how many teachers to wait for.

## Acceptance
- [ ] `scripts/train_seeds.py --task rababa_arabic_pro --seeds 42,1337,2026` runs all 3 in parallel.
- [ ] `distill_into_student` reduces DER vs single-seed by ≥ 5%.
- [ ] Pipeline stage `distill` integrates with `--skip-distill` flag.

## Files
- `scripts/train_seeds.py` (new)
- `src/rababa/training/distill.py` (new)
- `src/rababa/training/__init__.py` (export `distill_into_student`)
- `modal_app.py` (add `distill` function + stage)
- `tests/training/test_distill.py` (new)

## Open questions
- Should distillation use the same arwiki pretrain init as teachers,
  or train from scratch on gold + soft labels? Lean: same init.

# 03 — Noisy Student self-training

## Why
After supervised training, the model is confident-and-right on most
of arwiki (which has no gold labels). Use those high-confidence
predictions as silver labels, augment with input-side noise (char
dropout, keyboard confusables), retrain. Typical DER drop: 5-10%.

## Architecture

```
unlabeled arwiki ──→ trained model ──→ confidence filter
                                          │
                                          ↓ high-conf silver
                                      augment (noise)
                                          │
                                          ↓
            original gold ──→ + silver (noisy) ──→ retrain
```

## Tasks

### 3.1 Self-labeling function (`src/rababa/training/noisy_student.py`)
- `label_unlabeled(model, text_iter, batch_size, conf_threshold) -> list[Example]`
- Run inference on raw Arabic text, keep predictions where
  softmax_max > conf_threshold (default 0.95).
- Output is a list of `Example(input_ids=..., target_ids=..., raw=...)`,
  same shape as supervised examples.

### 3.2 Augmentation policy (`src/rababa/training/augment.py`)
- `CharDropout(p)`, `KeyboardConfusables(p)` as torch Dataset wrappers.
- Existing `scripts/clean_tashkeela_sadeed.py` already has iltiqā'
  rule; that's not augmentation per se, leave it.
- Augmentations applied via Compose at DataLoader-time, not pre-baked.

### 3.3 Noisy student loop
- `noisy_student_round(task, teacher_ckpt, unlabeled_path, n_rounds=1)`
- For each round:
  1. Label unlabeled text with teacher.
  2. Filter by confidence.
  3. Combine with original gold training set.
  4. Train a fresh student on combined set.
  5. New student becomes teacher for next round.

### 3.4 Modal function (`modal_app.py::noisy_student`)
- Wraps the loop with GPU + volume access.
- Reads arwiki from `/opt/rababa/data/arwiki/train.txt`.
- Writes augmented checkpoint to `/checkpoints/{task}/run-noisy/best.pt`.

## Acceptance
- [ ] `label_unlabeled` produces silver labels with >95% per-token confidence.
- [ ] `noisy_student_round` reduces DER vs baseline by ≥ 3% after 1 round.
- [ ] No regression on rare-haraqat classes (Shaddah+Kasratan, etc).

## Files
- `src/rababa/training/noisy_student.py` (new)
- `src/rababa/training/augment.py` (new)
- `modal_app.py` (add `noisy_student` function)
- `tests/training/test_noisy_student.py` (new)
- `tests/training/test_augment.py` (new)

## Open questions
- Confidence threshold: 0.95 might be too high. Sweep 0.85/0.90/0.95
  on val set during initial validation.

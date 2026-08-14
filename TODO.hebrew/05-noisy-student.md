# 05 — Noisy Student (Hebrew)

Same recipe as Arabic 03. Self-label hewiki unlabeled text with current
Hebrew model, filter by confidence, augment, retrain.

## Tasks

### 5.1 Label hewiki unlabeled text
- `label_unlabeled(model, hewiki_lines, ...)` from `training/noisy_student.py`
- Filter mean per-token confidence > 0.95

### 5.2 Combine with gold + augment
- `CombinedDataset(gold, silver, augment=default_hebrew_augment())`
- Hebrew augment = CharDropout only (no dot-variant confusables)

### 5.3 Retrain
- `noisy_student_round(task, teacher_ckpt, ...)` from `training/noisy_student.py`

## Acceptance
- [ ] 1 noisy-student round reduces DER by ≥ 2% vs baseline.

## Files
- (no new code — uses existing `training/noisy_student.py`)

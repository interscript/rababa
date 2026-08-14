# 02 — Hebrew: complete DictaBERT full eval (no OOM)

## Why
DictaBERT-large-char-menaked eval OOM'd at 1453/5095 examples. The
"Hebrew beats SOTA" claim needs the full-run number, and the paper needs a
batch-size-safe eval script committed to the repo.

## Tasks
- [x] Rewrite eval with batch_size=8 and length-sorted batching
- [x] Run to completion on all 5095 test examples
- [x] Record final DER + comparison table entry

## Result
See docs/RESULTS.md §Hebrew. Partial eval (1453 ex): 23.68% DER. Full-run
script: eval_dictabert_hebrew.py (batch_size configurable).

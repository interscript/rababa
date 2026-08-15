# 07 — Distillation runner

Purpose: bring non-byte models into the one-runtime world.
- ByT5-base -> ByT5-small logit distillation (case study: Hebrew s43;
  report DER before/after in RESULTS.md)
- Thai umt5 (sentencepiece, our 2.32% PER SOTA) -> ByT5-small student on
  the Thai corpus + epitran augmentation; NEVER ship a sentencepiece
  tokenizer into the runtimes
- Arabic char-encoder: export as single ONNX classifier + optional trie
  artifact (adapter, not distillation)
- A10G or queued A100; watchdog + resume; templates exist
  (train_khmer_byt5.py, train_arabic_byt5.py)

Acceptance: one distilled model shipped with before/after metrics.

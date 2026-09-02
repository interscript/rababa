# 06 — Python runtime: interscript-ml (pip)

- Thin onnxruntime wrapper; bytes tokenizer; greedy + KV decode
- from interscript_ml import Model; Model.load('khm-latn-1.0.zip')
- Parity-tested against the same zips and golden JSONL
- Also the reference implementation the other APIs are diffed against

Acceptance: pip installable from the repo, tests pass in CI, identical
outputs on golden set.

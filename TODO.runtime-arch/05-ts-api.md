# 05 — TypeScript runtime: @interscript/ml

- onnxruntime-web + onnxruntime-node; tokenizer = TextEncoder (free)
- Same greedy + KV decode as Ruby; shared golden JSONL tests (jest)
- Model loading from URL or file; sha256 via crypto.subtle before load
- npm packages per model (@interscript/model-khm-latn) so jsDelivr can
  serve browser inference; runtime never bundles models
- Minimal playground example for interscript.org (paste text, pick model,
  see output) — the adoption surface

Acceptance: browser demo loads from jsDelivr, runs khm-latn, output
matches Ruby on the golden set.

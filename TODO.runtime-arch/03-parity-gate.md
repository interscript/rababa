# 03 — Parity + checksum gate (mandatory before any release)

1. Parity: ONNX greedy vs HF generate(greedy) on >=500 samples from each
   model's test split; CER delta < 0.2pp; report written into
   metadata.yaml (samples, cer_delta).
2. Integrity: sha256 of each .onnx recorded in metadata.yaml; every loader
   (Ruby/TS/Python) verifies on load and fails loudly — this also solves
   the corrupt-download failure mode seen with 1.1GB zips.
3. Cross-runtime: Ruby == TS == Python on 100 fixed strings per model
   (checked into a golden JSONL per model).

Acceptance: gate script runs headless in CI; no zip ships without it.

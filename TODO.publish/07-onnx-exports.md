# 07 — ONNX export for TS/interscript.org consumption

## Why
interscript.org (TS side) needs runnable artifacts. ByT5's byte tokenizer is
TextEncoder — no vocab file needed in TS, making it ideal.

## Tasks
- [ ] Export best checkpoints to ONNX (rababa export_onnx infrastructure)
- [ ] Verify onnxruntime can load + run them
- [ ] Document artifact paths + sizes in RESULTS.md

## Status
Deferred — requires optimum/onnx export tooling pass per model family.

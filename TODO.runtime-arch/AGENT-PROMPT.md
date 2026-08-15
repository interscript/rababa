# Agent prompt: interscript-ml deployment stack

You are building the streamlined training→usage pipeline for Interscript's
neural phonological layer: ONNX export/distillation + Ruby/TS/Python
inference APIs, shipping models in the Interscript Model Format (IMF v1).

Read TODO.runtime-arch/00..10 in this directory; each is a work order with
goal, steps, and acceptance criteria. Execute in numeric order.

## Context
- Repos: interscript/rababa (ar/he diacritization), rababa-farsi, rababa-urdu,
  secryst/secryst (Ruby gem; PR #44: pure-Ruby Byt5Onnx engine, byte
  tokenizer pad=0 EOS=1, greedy decode; export script
  scripts/export_onnx_byt5.py — opset 14 REQUIRED, the Ruby onnxruntime
  gem's bundled ORT is old). Model zips on Modal volume
  secryst-checkpoints:/khmer_byt5/. docs/RESULTS.md in each repo = ground
  truth for metrics and checkpoint paths.
- Byte-level (ByT5-family) models share ONE runtime: tokenizer = UTF-8
  bytes (Ruby String#bytes, TS TextEncoder, Python bytes) — no vocab files.
- Models: khm-latn (fp16 zip exists, unverified), urd-g2p, urd-diac,
  fas-g2p (v4/v5 runs in flight), heb-diac (ByT5-base s43 — fp16+int8),
  ara-diac (char-encoder classifier — single ONNX + optional trie),
  Thai umt5 (sentencepiece — distill to ByT5 student, do NOT ship spm).

## Hard rules (non-negotiable, from the user)
- Feature branches + PRs only; NEVER push tags or to main; never
  `git add -A` — explicit paths, verify staged set; NO AI attribution;
  PR bodies via `--body-file` (inline backticks execute in shells);
  never delete files you didn't create; ASK before pushing anything.
- Modal: ALWAYS `modal run --detach`; long jobs wrapped in
  `until modal run --detach X; do sleep 60; done` watchdogs; scripts
  must checkpoint periodically + auto-resume (server evictions happen).
- Resource discipline: one A100 job at a time; exports/parity/distillation
  on A10G or CPU.

## Acceptance (overall)
- Ruby/TS/Python produce IDENTICAL outputs on the same model.zip for 100
  strings per model.
- Every shipped zip passes parity (ONNX vs HF greedy, CER delta <0.2pp on
  >=500 test samples) + sha256 verification on load.
- Releases PROPOSED with a manifest table (model, task, size fp16/int8,
  metric + protocol, source repo) — never tagged without the user.
- All work positioned as "interscript-ml — the phonological layer of
  Interscript"; rababa/secryst credited as component labs.

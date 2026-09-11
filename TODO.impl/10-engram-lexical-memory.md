# 10 — Engram-style lexical memory for diacritization (P3, gated)

Source: TODO.qwen-next/10 §6. The one architecture-level lever the
DeepSeek pass surfaced that fits our open question — the residual is
interaction-level and the corpus axis is closed, so architecture is
the remaining category. Haraqat are heavily lexical/n-gram driven;
a hashed n-gram lookup table is a direct fit for the idiom memory a
300M byte model lacks.

## Why gated

- Changes export and runtime: gather ops over a table inside the ONNX
  graph, table shipped in the zip. Browser size caps the table —
  int8 4M-entry x 64-dim (~64 MB) is the ceiling before the lite
  tier's size story breaks.
- Sequencing: only worth building if 02 (teacher mixture) and 04
  (trained lite) also fail to move the frontier — otherwise it
  spends the architecture budget before the cheap cells are closed.
- Corrected mechanism per 09: hash-addressed tables in the forward
  pass (DeepSeek config), not example retrieval.

## If built

- Table as a parameter module (addresses = n-gram hashes of the input
  bytes; embeddings summed at the two chosen layers); trained with
  the distill recipe; Sinkhorn-balanced table updates (06) are the
  natural pairing, as in the report.
- Export probe first: a table-augmented student exported to ONNX with
  decode-health smoke before any quality run.

## Acceptance (only after the gate)

- [ ] Size/latency budget for the browser tier documented
- [ ] Table module + export probe
- [ ] One run, full-set verdict with intervals

## Status

- [ ] Gated behind 02/04 results; not built

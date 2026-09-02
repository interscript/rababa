# 04 — Speculative decoding probe (PARKED)

Source: LongCat-Flash-Lite converts embedding sparsity into inference
speed via speculative decoding; Qwen ships day-0 vLLM support.

Ours: byte-level outputs are long (1400B window → up to 3200 tokens), so
the theory applies — a tiny byte draft model + the student as verifier
could cut server-tier batch decode latency.

Why parked:
- Teachers run offline; the API serves greedy small students that are
  already fast for their workloads.
- We have no latency-budget data showing decode time binds at the API.
- The draft model would itself need training + a parity story — not free.

Revisit trigger: API latency metrics showing p95 decode > budget, or a
client-tier memory-layer win (TODO 02) that inflates compute enough for
draft/verify to matter. Registered here so the idea isn't lost.

## Status

- [x] Parked with explicit revisit trigger (no code, by design)

# 11 — Static activation scales: fix decode framing at the source (P0)

Pre-registered 2026-09-12, from the framing finding (RESULTS.md:
dynamic-int8 graphs compute activation quantization scales per fed
tensor, so single-step and batched decode framings produce materially
different decodes). The shipped int8 uses `quantize_dynamic`
(DynamicQuantizeLinear at runtime); static quantization
(`quantize_static` + calibration) bakes activation scales into the
graph and removes the per-run computation entirely.

## Why this is the top lever

If framing stabilizes, three separate findings improve at once:
- the speculative tier's acceptance collapse (0.99 → 0.46 purely by
  framing) may reverse — its domain would reopen on CPU
- the cross-hardware near-tie flips (RESULTS 2026-09-07) should shrink:
  one fewer runtime-computed quantity to diverge across machines
- the golden test could return to byte-exact assertions for int8

## Experiment (pre-registered gates)

`scripts/static_int8_experiment.py`: re-quantize the 2.1 decoder-kv
graph with QUInt8 static activations (MatMul-only, head fp32 — same
node policy as the shipped dynamic int8), calibrated on 365 real decode
feeds sampled across BOTH framings (prefills, incremental steps,
8-token windows).

- **A. framing equality**: the same greedy decode driven token-by-token
  vs in 8-token batched calls — GATE: identical trajectories
- **B. drift vs fp32** (quality proxy): GATE: no worse than the
  dynamic int8 artifact's relationship to fp32; full-set DER via the
  artifact scorer if A clears
- **C. speed**: static vs dynamic tok/s on CPU — record, not gate
  (u8s8 static kernels are usually comparable or faster)

If A and B clear: re-export all quantized artifacts through the static
path (modal_export gains the calibration stage), index-v6, golden rows
regenerate, framing claim added to the contract.

## Status

- [x] Pre-registered (this file)
- [x] fp32 artifact fetched, sha-verified against index-v5
- [x] Calibration corpus collected (365 feeds, both framings)
- [ ] quantize_static result measured (A/B/C)
- [ ] Verdict recorded in RESULTS.md; ledger cross-links updated

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

## Verdict (2026-09-12, final): framing gate FAILED for the node fix — the instability is the node build itself

The test run at every endpoint (RESULTS.md, PR #214):

- Python ORT: 480/480 framing-stable at fp32, dynamic int8, AND
  static int8
- onnxruntime-node: 89-101/485 UNSTABLE at every precision INCLUDING
  fp32 — batch-shape-dependent kernel numerics before quantization
  enters; static scales cannot repair what is not a scale computation
- static int8 byproduct: quality-clean on sample (0/480 drift vs
  fp32) and ~8% faster than dynamic on CPU — stays a candidate for
  the export path on its own merits, gated by full-set DER, NOT by
  framing

Re-scoped conclusion: framing stability is a property of the runtime
distribution. Shipped TS paths are single-framing by construction, so
published byte parity is unaffected; framing is the parity variable
of mixed-shape serving (why the speculative tier degraded and was
pulled — TODO.impl/01).

## Status

- [x] Pre-registered (this file)
- [x] fp32 artifact fetched, sha-verified against index-v5
- [x] Calibration corpus collected (365 feeds, both framings)
- [x] quantize_static measured (A/B/C) across BOTH runtime builds
- [x] Verdict recorded in RESULTS.md + Paper C 5.1; framing-fix
      closed negative; static-int8 speed/quality kept open

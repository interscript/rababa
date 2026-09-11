# 05 — int8 KV cache in the decode path (P2, contract-gated)

Source: TODO.qwen-next/10 §5. DeepSeek's cache precision hierarchy —
main KV at FP4, the sensitive SWA branch at FP8, head fp32 — matches
our head-fp32 discipline and points at the one runtime tensor we
still keep at fp32: the decode-path KV cache. Caching K/V at int8
with per-head scales halves decode memory bandwidth; in the browser,
bandwidth is the binding cost.

## Design constraints

- This changes the IMF artifact surface (decoder-kv.onnx past/present
  dtypes, or scale tensors beside them) — it is an IMF v1 → v2
  question, not a silent tweak. All three runtimes (TS, Python,
  Ruby) must load both old and new zips, or the index pins a
  minimum-runtime version.
- The claim is quality-level parity for quantized artifacts (the
  golden-v1 scoping): gate with per-model cer_delta + decode-health
  smoke, NOT byte parity — the int8-weights story already established
  that contract.
- Keep head fp32 (unchanged discipline).

## Deliverables

1. Export-side experiment: one model exported with int8 KV pasts +
   per-head scales; measure cer_delta and decode speed on the
   artifact scorer.
2. If delta clears the existing quantized gates: spec the zip
   manifest fields (`kv_quant: int8-head`), runtime loader support
   behind a capability flag, and an index/runtime pin plan.
3. Browser wall-clock measurement before/after on the playground
   tier (the 95 MB lite model first).

## Acceptance

- [ ] Experiment artifact scored; cer_delta recorded
- [ ] Speed measurement recorded (decode tokens/sec, before/after)
- [ ] Contract decision documented (v2 fields or reject) — owner gate
- [ ] If adopted: all-runtime loading + golden rows smoke

## Status

- [ ] Awaiting 01/02 results before spending export effort; spec only

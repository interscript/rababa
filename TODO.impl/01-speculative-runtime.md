# 01 — Speculative decoding across the tier ladder (P0)

Source: TODO.qwen-next/10 §1 (unparks TODO.qwen-next/04). The parked
objection — "the draft model would itself need training" — expired
when ara-diac-layerdrop-1.0-int4 (190M, 95 MB) shipped: drafter and
verifier (ara-diac-small-2.1-int8, 300M) share the byte table and
greedy decode, and greedy verification is output-preserving, so the
artifacts need no parity story — only a speed one.

## Why

Byte-level decode is one token per step; the browser tier pays it in
wall-clock. DSpark's shape (draft K, verify all K in one pass,
confidence-scheduled K) maps directly onto two shipped models. Early
probe rows: acceptance 0.98–1.00, 8.5–8.9 tokens per verifier pass at
K=8 — the int4 drafter matches the int8 verifier's argmax almost
everywhere, consistent with the 0.036% near-tie flip measurement.

## Deliverables

1. **Acceptance probe** (CPU, golden-v1 Arabic rows): per-row
   acceptance rate, tokens/verify, exactness of the speculative loop
   vs the verifier's plain-path greedy (same execution path — the
   theorem check). Script:
   `ml-models/scripts/probe_speculative.py`, results to
   `~/ml-logs/spec_probe/results.json`.
2. **Runtime implementation** (interscript-ts): a
   `SpeculativeSession` composing drafter + verifier IMFModel
   instances — OCP: `translate()` untouched, the strategy is a new
   class; block size is config data, not code. Unit tests on the
   acceptance algorithm (synthetic logits are data, not model mocks);
   e2e opt-in with real zips.
3. **Tier decision**: if full-probe acceptance clears ~0.9, expose
   "2.1 quality at lite-ish latency" as a runtime option (not a new
   artifact). Publish the acceptance measurement next to the tier in
   docs.

## Acceptance

- [ ] Probe: 25 rows measured; exactness holds on every checked row
      (plain-path reference; long rows may skip the O(T^2) reference,
      recorded as unchecked)
- [ ] Probe bug class closed: EOS-block acceptance appends pre-EOS
      tokens; bonus index computed post-extend
- [ ] TS `SpeculativeSession` with unit + opt-in e2e tests, PR merged
- [ ] Numbers recorded in TODO.qwen-next/10 §1 and RESULTS.md if the
      tier ships
- [ ] Probe script landed in ml-models with its results

## Status

- [x] Probe v1 written; EOS-acceptance and index bugs found and fixed
- [ ] Probe rerun to completion
- [ ] Runtime implementation
- [ ] Tier exposure decision

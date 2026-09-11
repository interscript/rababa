# Per-Head Muon wire-in

## Status (2026-09-11)
Implemented in ml-models (this note was stale — no module had shipped
in v0.5.0): `src/gpu/muon.py` carries the headwise group
(`add_headwise_group`, per-slice Newton-Schulz, `qk_named` selector),
config-gated by the `headwise_muon` spec flag, off by default. Specs:
`ml-models/tests/test_muon_headwise.py` (5 tests, green) — per-head
update equals vanilla Muon on each head slice; heads=1 equals vanilla
whole; off-state routing unchanged. Externally validated by
DeepSeek-V4.1-Flash sec 2.5, GLM-5, Kimi-K3 (TODO.impl/03).
Remaining: exercise on the next teacher run (flag on).

## Acceptance
- [x] Feature is enabled by config flag, off by default
- [x] Spec covers the wire-in dispatch (tests + flag wiring)
- [ ] End-to-end training run completes with feature enabled
- [x] No regression on baseline (feature off → identical results)

## Files
- (TBD based on feature)

# N-stream mHC wire-in

## Status
Module is built (see v0.5.0 shipped). Wire-in needs:
- Config flag in YAML
- Dispatch in training loop or model builder
- End-to-end smoke test
- Spec for the wire-in

## Acceptance
- [ ] Feature is enabled by config flag, off by default
- [ ] Spec covers the wire-in dispatch
- [ ] End-to-end training run completes with feature enabled
- [ ] No regression on baseline (feature off → identical results)

## Files
- (TBD based on feature)

## Deployable variant (added 2026-09-11, DeepSeek-V4.1-Flash §2.4.1)

Single-Pass mHC: consume the PREVIOUS block's input-mixing coefficients
(A_{l-1}) instead of the current block's. This removes the reduction
dependency so residual update, input mixing, and coefficient prediction
fuse into one kernel — halving activation memory traffic at "negligible
degradation." If n-stream mHC resumes, train with the shift from the
start so the network adapts to it (their pattern: architecture change
plus training-aware adaptation, never architecture change alone).

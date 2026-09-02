# 36 — Zero-centered RMSNorm Implementation Fix

## Problem

The current `ZeroCenteredRMSNorm` (Qwen3.5) is broken:
- `gamma` initialized to **0** in `__init__`
- Forward: `out = norm * gamma + x`
- At init (gamma=0): `out = 0 + x = x` — **no normalization happens**

This caused Hebrew v0.6.0 to NaN after ~9 epochs of supervised training:
gamma stayed ≈0 → no normalization → activations grew unbounded through MoE
→ router weights exploded to norm=131 (init ≈19) → NaN.

## Fix

Change `gamma` init from **0** to **1**:
- At init (gamma=1): `out = norm + x` — proper normalization PLUS residual bypass.
- During training, gamma adapts to control normalization strength.

This matches the Qwen3.5 paper's intent: zero-centered means **gamma represents
deviation from identity**, but identity here means "normalized + residual"
(not "raw input").

## Files

- `src/rababa/models/zero_centered_rmsnorm.py` — change `torch.zeros` → `torch.ones`.
- `tests/models/test_zero_centered_rmsnorm.py` — update tests:
  - `test_gamma_init_is_zero` → `test_gamma_init_is_one`
  - `test_forward_is_identity_at_init` → `test_forward_normalizes_at_init`
- `configs/rababa_*.yaml` — re-enable `norm_type: zero_centered` (optional,
  after stability verified).

## Acceptance

- All existing specs pass.
- New spec: gamma init=1.
- New spec: forward at init produces output with RMS ≈ sqrt(2) (norm + x).
- Hebrew v0.6.0 retrain with `zero_centered` doesn't NaN.

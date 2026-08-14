# 07 — LatentMoE FFN (Hebrew, K3)

Same as Arabic 07. Replace FFN with mixture-of-experts.

## Tasks

### 7.1 Implement LatentMoE module
- `src/rababa/models/moe.py` (already designed in Arabic TODO 07)
- Shared implementation across all languages

### 7.2 Wire into modern.py for Hebrew
- `arch: "modern_multi_head_moe"` dispatch in `models/base.py`
- `ffn_type: "moe"` flag in Hebrew config

### 7.3 Train + benchmark
- Compare MoE vs SwiGLU at same param budget
- Hebrew DER should improve on rare classes (sin head especially)

## Acceptance
- [ ] MoE adds ≤ 20% params to baseline
- [ ] Per-class DER on rare sin/dagesh combos improves by ≥ 5%

## Files
- `src/rababa/models/moe.py` (shared, new)
- `configs/rababa_hebrew_v0.5.0.yaml` (new)

# 07 — LatentMoE FFN (Kimi K3)

## Why
Kimi K3's LatentMoE routes tokens through K latent experts via a
gated router, doubling model capacity at ~1.2× inference FLOPs.
For our small-model regime (~40M params), this is a large quality
win.

## Architecture

```
hidden → router (linear → softmax over K experts) → top-2 routing
                                                      ↓
                          expert_0(hidden)  expert_1(hidden)  ... expert_K(hidden)
                                  ↓               ↓                    ↓
                                          weighted sum by router probs
                                                      ↓
                                                  next layer
```

LatentMoE (vs traditional MoE) compresses expert weights through a
low-rank bottleneck so each expert is ~5% extra params, not 100%.

## Tasks

### 7.1 MoE layer (`src/rababa/models/moe.py`)
- `LatentMoE(dim, n_experts=4, expert_dim=None, top_k=2)`
- Router: `nn.Linear(dim, n_experts, bias=False)`.
- Each expert: low-rank Linear (up → gate → down) with rank << dim.
- Forward: compute router logits, top-k selection, gather + multiply.

### 7.2 ModernMoECharTransformer
- New arch variant: replace `w_gate/w_up/w_down` FFN in `ModernEncoderLayer`
  with optional `LatentMoE`.
- Triggered by `cfg.model.ffn_type: "moe" | "swiglu"` (default swiglu).

### 7.3 Load balancing loss
- Standard MoE auxiliary loss: encourage uniform routing distribution
  across experts.
- Added to total loss in supervised loop: `loss = task_loss + α · balance_loss`.

### 7.4 Wire into configs
- `configs/rababa_arabic_pro_v0.5.0.yaml`: `ffn_type: moe`, `n_experts: 4`.

## Acceptance
- [ ] `LatentMoE` with 4 experts adds ≤ 20% params to a 6L/512d model.
- [ ] Training loss converges; load-balance loss stays ≤ 0.05.
- [ ] Single-model MoE ≥ non-MoE at same param budget.

## Files
- `src/rababa/models/moe.py` (new)
- `src/rababa/models/modern.py` (add `ffn_type` param)
- `src/rababa/training/supervised.py` (add aux loss)
- `configs/rababa_arabic_pro_v0.5.0.yaml` (new)
- `tests/models/test_moe.py` (new)

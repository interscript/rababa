# Hebrew Diacritization — SOTA Implementation Map

> **Purpose**: Trace every modern SOTA technique applied to the Hebrew
> diacritization pipeline (rababa). Same encoder body as Arabic, with
> a multi-head output (niqqud, dagesh, sin) matching the legacy Nakdimon
> ONNX I/O contract.

## Models covered

| Model | Config | Role |
|-------|--------|------|
| `rababa_hebrew` | `rababa_hebrew.yaml` | Multi-head student (v0.6.0: 6L/384d, MoE, ~24M params) |
| `rababa_hebrew_pretrain` | `rababa_hebrew_pretrain.yaml` | MTP pretrain |
| `rababa_hebrew_dsv4` | `rababa_hebrew_dsv4.yaml` | + DS-V4-Flash Tier 1 (SwiGLU clamp, attn sink, sqrt_softplus, hybrid NS) |
| `rababa_hebrew_resformer` | `rababa_hebrew_resformer.yaml` | + DS-V4 + ResFormer + Muon variants (full 2026 stack) |
| `rababa_hebrew_resformer_only` | `rababa_hebrew_resformer_only.yaml` | + ResFormer ablation (no DS-V4) |
| `rababa_hebrew_adamuon` | `rababa_hebrew_adamuon.yaml` | + AdaMuon + NorMuon ablation (no architectural changes) |
| `rababa_hebrew_resformer_reg` | `rababa_hebrew_resformer_reg.yaml` | + ResFormer with stronger regularization (small-data fix) |

## Output contract (legacy compat)

Three independent softmax heads per character position, in canonical
order `OUTPUT_ORDER`:

| Head | Vocab | Role |
|------|-------|------|
| `niqqud` | 16 | Vowel points (holam, qamats, patah, etc.) |
| `dagesh` | 3 | Consonantal strengthening (none, mappiq, dagesh) |
| `sin` | 4 | Sin/shin dot position |

The legacy 2021 Nakdimon ONNX uses the same shapes — Ruby/TS runtime
needs no change when swapping in this model.

## SOTA techniques applied

The Hebrew student shares the encoder body with the Arabic student;
all techniques from [`IMPLEMENTATION_arabic.md`](IMPLEMENTATION_arabic.md)
apply here. The list below highlights Hebrew-specific decisions.

### 1. Multi-head output architecture

- **Code**: `ModernMultiHeadCharTransformer` (in `models/modern.py`).
- **Rationale**: Independent heads for each output category let the
  model learn niqqud, dagesh, and sin distributions separately.
  Empirically better than a single fused output head.
- **Encoder-shared**: Same encoder body serves all three heads; only
  the linear projection differs.

### 2. Same Qwen3 v0.6.0 stack as Arabic

- GQA: 6 query heads, 2 KV heads (3:1 group size).
- QK-Norm: stabilizes attention logits.
- Zero-centered RMSNorm: better gradient flow.
- ABF (rope_base: 1M): long-context extrapolation.
- Fine-grained MoE: 8 experts, top-2 activated, no shared experts
  (smaller than Arabic's 16 experts — Hebrew has 1/3 the parameter
  budget).
- MTP pretrain objective: 2-token prediction.

### 3. Distillation from Dicta (Tier 1.5)

- **Source**: Dicta's strong Hebrew diacritizer (CC-BY-NC).
- **Constraint**: Distillation only — we do not redistribute Dicta's
  weights, only the soft targets they produce on our training set.
- **Code**: `src/rababa/training/distill.py`.
- **Why**: Hebrew lacks large open diacritized corpora; distillation
  from Dicta injects signal equivalent to ~5x more labeled data.

### 4. Muon + QK-Clip

Same optimizer stack as Arabic. Smaller dim (384 vs 512) → smaller
`muon_lr` may be needed in practice (currently shared: 0.02).

### 5. 2026 cross-model techniques (v0.7.0)

Layered on top of the v0.6.0 stack. See `docs/CROSS_MODEL_2026_analysis.md`
for the full survey; key additions:

- **ResFormer** (arXiv:2410.17897, ACL 2025): value residual
  `V_n = λ_1·V_1 + λ_2·V_n` before attention. Sparse mode (last 2 layers,
  λ_1=5.0) per paper Table 3.
- **Spectral Cap Muon** (2026): Frobenius-norm cap on orthogonalized
  updates — direct fix for Hebrew NaN explosion history.
- **HTMuon** (arXiv:2603.10067, ACL 2026): heavy-tail α-blend with raw
  momentum. Better generalization on small datasets.
- **AdaMuon** (arXiv:2507.11005): element-wise second-moment estimator
  on orthogonalized updates (Adam-style adaptivity on orthogonal projection).
- **NorMuon** (arXiv:2510.05491): neuron-wise adaptive scaling. Fixes
  per-neuron non-uniformity in Muon updates.

**Known issue on Hebrew**: v0.6.0 + DS-V4 stack overfits (train_loss 1.57
vs val_loss 6.48 — 4.9 gap). Hebrew is small (~29K train pairs). The
`rababa_hebrew_resformer_reg` variant tests if stronger regularization
(3x dropout, 5x weight_decay, 2x label_smoothing, milder λ_1=1.0)
unlocks the technique's gain on small data.

## Training pipeline

```
[ Nakdimon corpus (Hebrew) ]
            ↓
[ Dicta distillation soft-targets ]
            ↓
[ MTP pretrain ]
            ↓
[ Supervised fine-tune (multi-head CE) ]
            ↓
[ Distill from Dicta (KL on soft targets) ]
            ↓
[ Multi-seed ×3 → ensemble → distill ]
            ↓
[ Noisy student self-training ×1-2 rounds ]
            ↓
        [ Ship ]
```

## Acceptance gates

| Version | Max DER | Notes |
|---------|---------|-------|
| v0.1.0 | 0.12 | First Hebrew student (shipped 2026-07-30) |
| v0.5.0 | 0.08 | With DS4/K3 modern stack |
| v1.0.0 | 0.06 | With Qwen3.5 stack + distillation |

## Operational notes

- Hebrew `best.pt` was 100% NaN after the SK-fix pretrain (2026-07-28).
  Force-retrained from scratch with the log-domain SK fix; subsequent
  runs are stable.
- All four artifacts (best.pt, fp32.onnx, q8.onnx, tflite) ship to the
  `/models` Modal volume. See `distribution-plan.md` memory entry for
  the GH Releases / HF / jsdelivr channel split.

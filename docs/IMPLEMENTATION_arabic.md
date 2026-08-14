# Arabic Diacritization — SOTA Implementation Map

> **Purpose**: Trace every modern SOTA technique applied to the Arabic
> diacritization pipeline (rababa). Each entry includes the paper
> reference, the rationale for adopting it, the integration point in our
> code, the config flag that activates it, and the expected impact on
> DER (Diacritization Error Rate).

## Models covered

| Model | Config | Role |
|-------|--------|------|
| `rababa_arabic` | `rababa_arabic.yaml` | Baseline (legacy 6L/384d, ~5M params) |
| `rababa_arabic_pretrain` | `rababa_arabic_pretrain.yaml` | MLM pretrain for baseline |
| `rababa_arabic_pro` | `rababa_arabic_pro.yaml` | SOTA student (v0.6.0: 6L/512d, MoE, ~80M params) |
| `rababa_arabic_pro_pretrain` | `rababa_arabic_pro_pretrain.yaml` | MTP pretrain for Pro |

## SOTA techniques applied

### 1. RoPE — Rotary Positional Embeddings

- **Paper**: Su et al. 2021 (arXiv:2104.09864).
- **Rationale**: Replaces learned absolute positional embeddings.
  Generalizes to longer sequences at inference; standard in DS4/K3/Qwen3.
- **Code**: `src/rababa/models/modern.py:RotaryEmbedding`,
  `apply_rope`.
- **Config**: `model.rope_base` (default 10000.0).
- **ABF (Qwen3-S3)**: Bump `rope_base: 1000000.0` for long-context
  extrapolation. Active in v0.6.0 configs.

### 2. SDPA — Scaled Dot-Product Attention

- **API**: `torch.nn.functional.scaled_dot_product_attention`.
- **Rationale**: Auto-triggers Flash Attention / memory-efficient kernels
  in PyTorch 2.x. No code change needed beyond calling the API.
- **Code**: `ModernEncoderLayer._attention`.

### 3. mHC — Manifold-Constrained Hyper-Connections

- **Paper**: DeepSeek V4 (arXiv:2512.24880).
- **Rationale**: Replaces the standard `x + sublayer(x)` residual with a
  learned SK-normalized 2×2 mixing matrix. The Sinkhorn-Knopp projection
  onto the Birkhoff polytope guarantees the residual stream doesn't
  collapse during from-scratch pretraining.
- **Code**: `src/rababa/models/modern.py:MHC` (2-stream),
  `MHCN` (N-stream generalization for decoder layers).
- **Numerical stability fix**: log-domain Sinkhorn (`sinkhorn_knopp`)
  avoids division-by-near-zero failures that the direct formulation
  produced for ~10% of random inits.

### 4. AttnRes — Attention Residuals

- **Paper**: Kimi K3 (arXiv:2607.24653).
- **Rationale**: Each layer's attention output is added to the next
  layer's attention input. Improves information flow across depth.
- **Code**: `ModernEncoderLayer.forward` — `prev_attn` threaded
  through layer iterations in `ModernCharTransformer.forward_encoder`.

### 5. RMSNorm

- **Standard in**: DS4, K3, Llama, Qwen3.
- **Rationale**: Drops LayerNorm's mean-centering; cheaper, empirically
  equivalent or better.
- **Code**: `src/rababa/models/modern.py:RMSNorm`.

### 6. Zero-Centered RMSNorm (Qwen3.5)

- **Paper**: Qwen3.5 (2026).
- **Math**: `out = x * rsqrt(mean(x²)+eps) * gamma + x`, gamma init=0.
  Identity at init; training learns gamma as deviations from identity.
  Better gradient signal for deep stacks.
- **Code**: `src/rababa/models/zero_centered_rmsnorm.py`.
- **Config**: `model.norm_type: "zero_centered"`. Default in v0.6.0
  configs.
- **Specs**: `tests/models/test_zero_centered_rmsnorm.py`.

### 7. SwiGLU FFN

- **Standard in**: Llama, DS, Kimi.
- **Math**: `w_down(silu(w_gate(x)) * w_up(x))`.
- **Code**: `ModernEncoderLayer._ffn` (default branch).

### 8. LatentMoE — Low-Rank Mixture of Experts

- **Paper**: Kimi K3 (arXiv:2607.24653) for low-rank experts;
  Qwen3 (arXiv:2505.09388) for fine-grained recipe.
- **Rationale**: Top-K routed MoE doubles model capacity at ~1.2x
  inference cost. Qwen3 fine-grained recipe: many small experts (32+),
  no shared experts, global-batch load balancing.
- **Code**: `src/rababa/models/moe.py:LatentMoE`, `LatentExpert`.
- **Config**:
  ```yaml
  model:
    ffn_type: moe
    moe:
      n_experts: 16         # fine-grained for Pro scale
      expert_dim: 512       # low-rank (vs FFN's 2048)
      top_k: 2
      shared_experts: 0     # Qwen3: no shared experts
  ```
- **Load balance loss**: `LatentMoE.load_balance_loss(global_batch=True)`
  is the Qwen3 recipe. Wired into supervised + MLM + MTP loops via
  `_collect_moe_lb()` helper. Weight: `train.moe_lb_weight: 0.01`.

### 9. GQA — Grouped Query Attention

- **Paper**: Ainslie et al. 2023 (Llama-2); Qwen3 standard.
- **Rationale**: KV heads < query heads → smaller KV cache, faster
  attention, no quality loss at typical ratios (4:1 to 8:1).
- **Code**: `ModernEncoderLayer.__init__` builds `q_proj` + `kv_proj`
  when `kv_heads < heads`; `_attention` does `repeat_interleave` on
  the group dimension.
- **Config**: `model.kv_heads: 2` (with `heads: 8`, ratio 4:1).

### 10. QK-Norm

- **Paper**: Qwen-Max / Gemma-2.
- **Rationale**: RMSNorm on Q and K vectors before the attention dot
  product stabilizes logit magnitudes. Pairs with QK-Clip (Mu) for
  deep-stack training.
- **Code**: `ModernEncoderLayer.{q_norm, k_norm}` (RMSNorm on `head_dim`).
- **Config**: `model.qk_norm: true`.

### 11. KDA — Kimi Delta Attention

- **Paper**: Kimi K3 (arXiv:2607.24653).
- **Rationale**: Per-layer learnable scalar attention bias. Cheap,
  empirically helps long-context tasks.
- **Code**: `src/rababa/models/kda.py:KDABias`, `softmax_with_kda`.
- **Config**: `model.kda: true`. Currently OFF in v0.6.0 (we already
  have QK-Norm; enable if logit-magnitude drift appears).

### 12. Muon Optimizer + Per-Head Muon

- **Paper**: Karpathy / Moonshot (K3).
- **Math**: Newton-Schulz orthogonalization of 2D weight gradients.
  1D params + embeddings use AdamW.
- **Code**: `src/rababa/training/optim.py:MuonAdamWHybrid`,
  `src/rababa/training/per_head_muon.py:PerHeadMuon`.
- **Per-Head Muon**: Orthogonalizes QKV/out_proj gradients
  per-head-slice rather than as a whole matrix. Better conditioning
  for multi-head attention.
- **Config**:
  ```yaml
  train:
    optimizer: muon
    muon_lr: 0.02
    muon_momentum: 0.95
    ns_steps: 5
    use_per_head_muon: true   # optional
  ```

### 13. QK-Clip

- **Paper**: Kimi K2 MuonClip (arXiv:2502.20776).
- **Rationale**: Anneals attention-logit bound tau from 8 → 1 over
  training. Prevents logit explosion that destabilizes deep training.
- **Code**: `src/rababa/training/optim.py` (clip hook).
- **Config**:
  ```yaml
  train:
    qk_clip_every: 50
    qk_clip_tau_init: 8.0
    qk_clip_tau_final: 1.0
  ```

### 14. MTP — Multi-Token Prediction

- **Paper**: DeepSeek V4 (arXiv:2512.24880) — pretraining objective.
- **Math**: N parallel prediction heads per position. Per-token CE
  with geometric weights `1/sqrt(i+1)`. ~1.5x more sample-efficient
  than MLM.
- **Code**: `src/rababa/models/mtp.py:MTPHead`, `mtp_loss`;
  `src/rababa/training/pretrain_mtp.py:pretrain_mtp`.
- **Config**:
  ```yaml
  train:
    pretrain_method: mtp   # mlm | electra | mtp
    mtp_n_predict: 2
  ```
- **Use**: PRETRAINING ONLY. Heads discarded before supervised
  fine-tune.

### 15. ELECTRA — Replaced Token Detection

- **Paper**: Clark et al. ICLR 2020.
- **Rationale**: Discriminator predicts per-position "original or
  replaced". Trains on ALL positions (vs MLM's ~15%) → ~2x more
  sample-efficient.
- **Code**: `src/rababa/training/electra.py`.
- **Config**: `train.pretrain_method: electra`.

### 16. Curriculum Learning

- **Paper**: Hacivia et al. 2009; Bengio et al.
- **Rationale**: Order training examples by difficulty. Easier examples
  first → faster convergence.
- **Code**: `src/rababa/training/curriculum.py:CurriculumSampler`,
  `src/rababa/features/arabic.py:compute_arabic_features` (difficulty
  signals: iltiqaa_violation, word_boundary, consonant_class).

### 17. Multi-Task Heads

- **Heads**: `output` (haraqat) + `seg` (word boundary).
- **Rationale**: Word-segmentation labels are trivially derived from
  input (1 after each space). Regularizes the encoder with free signal.
- **Code**: `ModernCharTransformer.seg_head`.
- **Config**: `model.with_seg_head: true`.

### 18. Trie-Constrained Decoding

- **Rationale**: Forces haraqat output to only emit sequences that
  appear in a reference lexicon (validated haraqat combinations per
  consonant). Hard constraint; turns impossible outputs into valid ones.
- **Code**: `src/rababa/decoding/trie.py` (lexicon builder + trie beam).

### 19. Multi-Seed Ensemble + Distillation

- **Rationale**: Train N models with different seeds; distill into a
  single student via KL on temperature-scaled softmax. Typical DER
  improvement: 5–10%.
- **Code**: `src/rababa/training/multi_seed.py`,
  `src/rababa/training/distill.py`.

### 20. Noisy Student Self-Training

- **Paper**: Xie et al. 2020 (Noisy Student).
- **Rationale**: Train teacher → self-label unlabeled data with
  augmentation → retrain student on combined labeled + self-labeled.
- **Code**: `src/rababa/training/noisy_student.py`.
- **Constraint**: NO LLM teacher. Only self-generated labels.

### 21. Engram — Episodic Memory

- **Paper**: DS4.
- **Rationale**: Per-layer episodic buffer of (hidden, label) pairs.
  Cosine retrieval of top-K similar hidden states → gated mix into
  current hidden. Helps rare patterns.
- **Code**: `src/rababa/models/engram.py:Engram`.

## Training pipeline

```
[ Tashkeela + Sadeed + QCRI corpus ]
                ↓
        [ Sadeed-style cleaner ]
                ↓
[ MTP pretrain (Pro) or MLM pretrain (baseline) ]
                ↓  (encoder checkpoint)
        [ Supervised fine-tune ]
                ↓  (with seg head, Muon, QK-Clip)
        [ Tier-1 student ]
                ↓
[ Multi-seed ×3 → ensemble → distill ]
                ↓
[ Noisy student self-training ×1-2 rounds ]
                ↓
[ Trie-constrained beam at inference ]
                ↓
            [ Ship ]
```

## Acceptance gates (per version)

| Version | Max DER | Notes |
|---------|---------|-------|
| v0.5.0 | 0.04 | First Pro retrain |
| v1.0.0 | 0.025 | Sadeed-corrected territory |
| v1.5.0 | 0.018 | With Qwen3.5 stack |

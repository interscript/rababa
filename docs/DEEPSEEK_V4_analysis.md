# DeepSeek-V4-Flash Techniques Analysis — Applicable to Rababa/Secryst

> **Source**: DeepSeek-V4 paper (arXiv:2606.19348, April 2026).
> **Scope**: V4-Flash (284B total, 13B activated) — the smaller, more relevant variant
> for our sub-1B diacritization/G2P models. Same architecture as V4-Pro (1.6T/49B),
> different scale.
> **HF card**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

## Scale comparison

| Model | Total params | Activated | Context |
|---|---|---|---|
| DeepSeek-V4-Pro | 1.6T | 49B | 1M |
| DeepSeek-V4-Flash | 284B | 13B | 1M |
| Rababa ModernCharTransformer | ~5–10M | ~5–10M | 512 |
| Secryst ModernSeq2Seq | ~5–10M | ~5–10M | 256 |

We're 4 orders of magnitude smaller. The transferable techniques are **architectural
and optimization choices**, not the scale-dependent ones (CSA/HCA, FP4 QAT).

## Architecture summary (from paper + HF card)

V4-Flash shares the same three pillars as V4-Pro:

1. **Hybrid attention (CSA + HCA)** — only useful >8K context. **Skip** for us.
2. **mHC (Manifold-Constrained Hyper-Connections)** — already in `models/modern.py`.
3. **Muon optimizer** — already in `training/optim.py`.

The HF card's headline summary mentions only these three. Deeper techniques
(SwiGLU clamping, attention sink, partial RoPE, etc.) appear in the paper's
detailed sections.

## Two-stage post-training pipeline (HF card)

Interesting architectural decision — but largely orthogonal to our work:
1. Independent cultivation of domain-specific experts (SFT + RL/GRPO per domain).
2. Unified consolidation via on-policy distillation (OPD) — see Tier 3 below.

We don't do RL or domain-specific experts. The OPD idea is interesting but
needs a teacher trajectory pipeline we don't have. Defer.

## What we already have (validated by V4 paper)

These techniques we adopted based on V3/V4 leaks — confirmed in the paper:

- **mHC (Manifold-Constrained Hyper-Connections)** — Paper Section 2.2, our `models/modern.py:MHC, MHCN`
- **Muon Optimizer** — Paper Section 2.4, our `training/optim.py:MuonAdamWHybrid`
- **MTP (Multi-Token Prediction)** — Paper Section 2.1, our `models/mtp.py`
- **LatentMoE** — Paper Section 2.1, our `models/moe.py:LatentMoE` (we use LatentMoE from K3 which is similar to DeepSeekMoE)
- **RMSNorm** — Paper Section 2.3.3, our `models/modern.py:RMSNorm`

## New techniques from V4-Flash we should adopt

### Tier 1: Quick wins (high impact, low effort)

1. **SwiGLU Clamping** — Paper Section 4.2.3. Clamp linear component to [-10, 10],
   cap gate at 10. Eliminates outliers, stabilizes training.
   - **Why**: We hit NaN explosions from MoE router drift. This is a direct fix.
   - **Effort**: 5 min — one clamp call in `_ffn`.
   - **Status**: TODO (#241)

2. **Hybrid Newton-Schulz for Muon** — Paper Section 2.4. Two-stage coefficients:
   - Steps 1-8: (a,b,c) = (3.4445, -4.7750, 2.0315) — rapid convergence.
   - Steps 9-10: (a,b,c) = (2, -1.5, 0.5) — stabilize at exactly 1.
   - **Why**: Faster Muon convergence + better stability. Our current single-coefficient NS is suboptimal.
   - **Effort**: 10 min — modify `zeropower_via_newtonschulz5`.
   - **Status**: TODO (#242)

3. **Sqrt(Softplus(·)) MoE affinity** — Paper Section 2.1. Replaces sigmoid with
   `Sqrt(Softplus(·))` for routing affinity scores. Better gradient flow than softmax.
   - **Why**: Our router weights exploded to norm=131. This is a direct fix.
   - **Effort**: 5 min — change router activation.
   - **Status**: TODO (#243)

4. **Attention Sink** — Paper Section 2.3.3. Learnable per-head sink logits
   allow attention to "leak" mass. Prevents first-token overattention.
   - **Effort**: 15 min — add learnable `sink_logit` parameter to attention.
   - **Status**: TODO (#244)

5. **Q/K RMSNorm** — Paper Section 2.3.3. We have Q-Norm; need K-Norm too.
   "Effectively prevents attention logits from exploding."
   - **Effort**: 5 min — add `k_norm = RMSNorm(head_dim)` to QK-Norm.
   - **Status**: Already implemented (verified by `tests/models/test_v060_techniques.py:test_qk_norm_adds_norm_modules`). TODO is just to add a spec asserting K-Norm presence. (#245)

### Tier 2: Medium effort, high value

6. **Partial RoPE** — Paper Section 2.3.3. Apply RoPE only to last 64 dims, not all.
   - **Why**: Standard practice; better generalization at long contexts.
   - **Effort**: 15 min — modify `apply_rope`.

7. **Anticipatory Routing** — Paper Section 4.2.3. Use historical params θ_{t-Δt}
   for routing decisions. Breaks vicious cycle from outliers.
   - **Why**: Direct complement to our NaN auto-recovery. Triggers on loss spikes.
   - **Effort**: 30 min — cache router state from previous step.

8. **Auxiliary-loss-free MoE balancing** — Paper Section 2.1. Replace our
   load_balance_loss with bias-update-speed trick (much smaller weight: 0.0001).
   - **Effort**: 30 min — change load balance strategy.

9. **Hash routing for early MoE layers** — Paper Section 2.1. First 3 MoE layers
   use hash(token_id) → expert. Stabilizes early training.
   - **Effort**: 20 min — add `HashRouter` option to first N layers.

### Tier 3: Larger efforts (defer)

10. **CSA (Compressed Sparse Attention)** — Only worth it at >8K context.
    Our max_len is 256-512. Skip.

11. **HCA (Heavily Compressed Attention)** — Same; only for long context.

12. **Sliding Window Attention branch** — Useful for >2K context. We're below.

13. **FP4 Quantization-Aware Training** — Hardware-specific, requires MXFP4 support.

14. **OPD (On-Policy Distillation)** — Replace KL-on-softmax with reverse KL on
    full vocab from teacher trajectories. Better than our current distill.py.
    - **Effort**: 1-2 days — needs trajectory generation + full-vocab KL.
    - **Note**: The HF card highlights this as the second post-training stage.
      We have no teacher trajectory pipeline, so it's a heavy lift.

15. **Generative Reward Model** — For RL post-training. We don't do RL.

16. **Muon ZeRO hybrid** — Distributed training concern. Single-GPU for us.

## Implementation status (2026-08-08)

All Tier 1 techniques shipped with specs:

| Technique | Code | Spec | Status |
|---|---|---|---|
| SwiGLU Clamping | `models/swiglu.py`, `models/modern.py:_ffn`, `models/moe.py:LatentExpert` | `test_dsv4_flash_techniques.py:test_swiglu_*` (6 specs) | ✅ |
| Hybrid Newton-Schulz | `training/optim.py:zeropower_via_newtonschulz5` | `test_hsv4_flash_techniques.py:test_hybrid_ns_*` (5 specs) | ✅ |
| Sqrt(Softplus) affinity | `models/moe.py:_router_affinity` | `test_dsv4_flash_techniques.py:test_sqrt_softplus_*` (7 specs) | ✅ |
| Attention Sink | `models/modern.py:_attention_with_sink` | `test_dsv4_flash_techniques.py:test_attention_sink_*` (6 specs) | ✅ |
| Q/K RMSNorm | `models/modern.py:q_norm, k_norm` | `test_dsv4_flash_techniques.py:test_qk_norm_*` (3 specs) | ✅ (already present, just verified) |
| Muon RMS rescale (bonus) | `training/optim.py:Muon.__init__` | `test_dsv4_flash_techniques.py:test_muon_*` (2 specs) | ✅ |

**Test suite: 204 passing, 7 skipped (slow/GPU-required), 0 failures.**

Configs not yet updated to enable these — backward-compat defaults preserve existing
checkpoints. To enable on new training runs:

```yaml
model:
  swiglu_clamp_max: 10.0       # DS-V4-Flash §4.2.3
  use_sink: true               # DS-V4-Flash §2.3.3
  moe:
    affinity_type: sqrt_softplus  # DS-V4-Flash §2.1
    swiglu_clamp_max: 10.0
train:
  optimizer: muon
  ns_steps: 10                  # DS-V4-Flash uses 10 (8 aggressive + 2 stable)
  muon_update_rms_rescale: 0.18 # expose via build_optimizer if adopted
```

## Recommended action plan

**Implement now (Tier 1, ~45 min total):**
- SwiGLU Clamping (#241)
- Hybrid Newton-Schulz (#242)
- Sqrt(Softplus) MoE affinity (#243)
- Attention Sink (#244)
- Q/K RMSNorm verification (#245, K-Norm already present)

**Implement next (Tier 2, ~2-3h total):**
- Partial RoPE
- Anticipatory Routing
- Hash routing for early layers
- Auxiliary-loss-free MoE balancing

**Defer (Tier 3):**
- CSA/HCA (not relevant at our context length)
- OPD (post-training, not core)

## Expected impact

- **Stability**: SwiGLU clamping + anticipatory routing should eliminate the NaN
  explosions we saw in Hebrew v0.6.0 (zero-centered RMSNorm incident).
- **Convergence**: Hybrid Newton-Schulz typically gives 10-20% faster convergence.
- **MoE quality**: Sqrt(Softplus) + hash routing + noaux balancing should give
  better expert utilization (currently we see router weight explosion).
- **Memory**: Partial RoPE saves ~50% of RoPE compute.

## References

- Paper: https://arxiv.org/abs/2606.19348
- HTML: https://arxiv.org/html/2606.19348v1
- V4-Flash HF card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- V4-Pro HF card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- Code: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference

## Related paper analysis: arXiv:2607.27146 (MindForge)

The user asked to also analyze arXiv:2607.27146. It is **MindForge: Teaching
Small Language Models Whole-Life-Cycle Software Engineering via Source-Free
Program Synthesis** (Chen et al., 2026).

### What MindForge does

- Converts open-source CLI programs into "source-free environments" exposing
  only a compiled reference executable + its documentation.
- Curates program synthesis trajectories using **GLM-5.2 as teacher agent**.
- Fine-tunes Qwen3.6-27B on those trajectories.
- Improves ProgramBench from 37.98% → 49.51%.

### Relevance to Rababa/Secryst: **low**

The paper is about **coding agent training**, not architecture. The transferable
ideas for us are:

1. **Reinforces our no-LLM-teacher rule** (already in memory). MindForge uses
   GLM-5.2 as teacher with a **verifiable test runner** as ground truth. We have
   no such verifier for haraqat — LLM-generated labels would be unverifiable
   hallucinations. MindForge validates that LLM teachers require a domain
   verifier; we don't have one, so we must avoid LLM teachers.

2. **Trajectory distillation** — MindForge distills trajectories, not just
   final answers. This is essentially the OPD technique from V4-Flash. Both
   papers point at the same idea (on-policy trajectory distillation with
   teacher) as the frontier post-training recipe. We don't have a teacher
   pipeline, so this remains deferred.

3. **Whole-life-cycle data curation** — methodology for building training
   data covering all stages of a task. For us, the equivalent would be
   covering all morphological contexts of Arabic/Hebrew words. Already
   addressed by our existing corpus curation.

### Direct technique adoption: **none**

MindForge is a software engineering paper. We don't synthesize programs. The
only transferable idea is "use verifiable ground truth when distilling" — which
we already enforce.

### Citation

```
@misc{chen2026mindforge,
  title={MindForge: Teaching Small Language Models Whole-Life-Cycle Software
         Engineering via Source-Free Program Synthesis},
  author={Yihao Chen and Shi Chang and Khaled Chawa and Feng Lin and Boyuan Chen
          and Shaowei Wang and Ahmed E. Hassan},
  year={2026},
  eprint={2607.27146},
  archivePrefix={arXiv},
}
```

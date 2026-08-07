# DeepSeek V4 + Kimi K3 → applicability for rababa_arabic_pro

**Goal:** push rababa_arabic_pro (50M params, browser-deployable, char-level
Arabic diacritization) below SUKOUN's 1.11% and Sadeed's 1.2% DER on the
Fadel benchmark, by adopting techniques from the latest DeepSeek and Kimi
papers where they apply at our scale.

**Bottom line:** two techniques are clear wins, four are skippable at our
parameter count and context length, two are optional Tier-3 bets. The two
wins — **mHC** and **MuonClip + QK-Clip** — bake into the existing 1-week
sprint at zero wall-clock cost.

---

## DeepSeek V4 (arXiv:2606.19348, released April 24, 2026)

V4 family: V4-Pro 1.6T total / 49B active MoE, V4-Flash 284B / 13B active
MoE. MIT-licensed. Hybrid CSA + HCA attention. The V4 paper is a synthesis
of six months of DeepSeek component papers (mHC, Engram, DeepSeekMoE,
Muon, MTP, R1) — most of which were validated in V3 / V2 / standalone
arXiv releases first.

### Component matrix

| Component | Paper | Take? | Why |
|---|---|---|---|
| **mHC** (Manifold-Constrained Hyper-Connections) | `2512.24880` | **YES** | Stabilizes from-scratch pretrain |
| **Engram** (N-gram conditional memory) | `2601.07372` | no | Our trie decoder is functionally equivalent |
| DeepSeek Sparse Attention (1M-token) | inherited from V3.2 | no | We have 256-512 tokens |
| MLA (Multi-head Latent Attention) | V2 `2405.04434` | no | KV cache is ~5 MB at our scale |
| **DeepSeekMoE** + aux-loss-free balancing | V3 `2412.19437` | **YES if MoE** | Cleanest MoE in existence |
| MTP (multi-token prediction) | V3 | no | Non-autoregressive task |
| **Muon optimizer** (adopted from Kimi) | K2 `2507.20534` | **YES, biggest win** | 2× compute efficiency |
| R1 RL reasoning | `2501.12948` | no (v2 bet) | We have ground-truth labels |

### Take in detail

**1. mHC (Manifold-Constrained Hyper-Connections) — drop-in for the encoder block.**

Mechanism: multi-stream residual connections where the mixing matrix is
projected onto a doubly-stochastic manifold via Sinkhorn-Knopp iterations.
Replaces the standard `x + sublayer(x)` residual with a learned multi-stream
mix. V4-Flash-0731 uses expansion factor 4 with 20 SK iterations.

Why it matters for us: our 12L / 768d encoder is trained from scratch —
high instability risk. The mHC paper reports faster convergence and
improved training stability vs. standard residual connections. The
identity-guarantee from the manifold constraint prevents the residual
stream from collapsing during long pretraining runs.

Implementation: ~30 lines in `src/rabba/model.py`. Replace
`return x + sublayer(x)` with `return mHC([stream_a, stream_b], x,
sublayer(x))` where the mixing matrix is normalized via SK iterations
every forward pass.

Expected gain: **−5 to −10% relative DER** from being able to train
harder and longer without loss spikes.

**2. MuonClip + QK-Clip — single biggest win.**

Mechanism:
- **Muon** (Keller Jordan, late 2024): matrix-aware optimizer for 2D
  weight matrices. Uses Newton-Schulz iteration to orthogonalize the
  gradient update. Reported 2× compute efficiency vs AdamW on LLM
  pretraining (Essential AI scaling laws, Feb 2025), 10-15% fewer tokens
  to reach same loss.
- **QK-Clip** (Kimi K2's contribution): per-head clipping of attention
  logits above threshold τ≈8, annealed over training. After each forward
  pass, if `QK^T` values exceed τ, clip them. Prevents the
  attention-logit explosion that destroys from-scratch training.
  Reportedly enabled Kimi K2's "zero loss spike" 15.5T-token pretrain.
- **Hybrid optimizer**: Muon for 2D weight matrices (Linear, Embedding),
  AdamW for 1D params (LayerNorm, biases). This is the standard Muon
  deployment pattern.
- **Per-Head Muon** (Kimi K3 extension): each attention head's weights
  get independent Muon updates, rather than treating the whole QKV
  projection as one matrix.

Why it matters for us: from-scratch char-level pretraining is the most
spike-prone training we do. Muon halves the wall-clock cost of the
pretrain stage. QK-Clip eliminates wasted compute on restarts.

Implementation: ~150 lines total. `MuonOptimizer` class with Newton-Schulz
iterator. `QKClipHook` on attention layers, applied every N steps with
τ=8 (Kimi's reported value), gradually annealed. Per-head Muon needs
refactoring the QKV projection to treat heads independently.

Expected gain: **2× faster pretrain (3h instead of 6h on A100)** AND
lower final loss. Compounds with everything downstream.

**3. DeepSeekMoE + auxiliary-loss-free balancing — if we go MoE.**

Mechanism: V3's bias-update trick replaces the standard auxiliary
load-balancing loss. Per-expert bias is incremented when an expert is
over-routed, decremented when under-routed. Pure routing signal, zero
gradient pollution from an aux loss.

Why it matters for us: if we apply MoE to our FFN (optional Tier 3 in
the sprint plan), this is the cleanest balancing recipe available.
Switch Transformer's aux loss was known to interfere with the main
objective; DeepSeek V3's bias trick solves that.

Implementation: ~80 lines. Replace top-k routing with bias-adjusted
top-k. Track per-expert load, adjust bias at the end of each step.

Expected gain: **−5 to −10% relative DER at same inference cost** (MoE
itself gives the gain; the aux-loss-free trick just makes training
stable).

### Skip in detail

**4. Engram (arXiv:2601.07372) — skip, but only because we have a better
equivalent.**

Mechanism: N-gram-keyed lookup table → conditional memory. Static facts
stored on CPU RAM, looked up at inference by token n-grams. Zero GPU
memory cost. Designed for 100B+ models where every GB of VRAM matters.
Adds a new "sparsity axis" beyond MoE.

Why skip at our scale: our 50M model fits in 200 MB VRAM. The "external
lookup memory" axis is already covered by our **trie-constrained beam
decoder** (T1.1 in the sprint plan) — the undiacritized-word →
diacritized-form dictionary is functionally the same idea, just keyed at
the word level instead of N-gram level.

v2 bet: if the trie decoder alone doesn't get us to ≤1.0%, extend it to
char-level N-grams (Engram-style) as v2.

**5. MLA (Multi-head Latent Attention) — skip.**

Mechanism: KV cache compression via low-rank latent. DeepSeek V2 reduces
KV cache by ~93%. Designed for inference-time memory pressure on
long-context autoregressive models.

Why skip: KV cache for 512 tokens × 768 dim × 12 layers = ~5 MB. Not a
bottleneck. MLA's low-rank projection adds complexity for zero gain.

**6. NSA / DeepSeek Sparse Attention — skip.**

Mechanism: 3-branch sparse attention (compressed coarse / selected
fine / sliding window) for 64K-1M token contexts. ACL 2025 best paper.

Why skip: char-level sequences are 256-512 tokens. O(N²) attention is
microseconds. Sparse adds complexity for zero gain.

**7. MTP (multi-token prediction) — skip.**

Mechanism: train a small draft head to predict the next K tokens jointly,
then use it for speculative decoding at inference. DeepSeek V3 sets a
2-token MTP objective.

Why skip: MTP is designed for autoregressive generation speedup via
speculative decoding. Our model is one-shot per-char classification
(whole sequence encoded in parallel, all haraqat predicted
simultaneously). Doesn't apply.

**8. R1 RL reasoning — skip for v1, v2 bet.**

Mechanism: GRPO with verifiable rewards. R1-Zero shows pure RL (no SFT
cold start) can incentivize reasoning on top of a base model.

Why skip for v1: Arabic diacritization has ground-truth labels —
supervised learning is the correct paradigm. We don't need reasoning
chains; we need per-char classification.

v2 bet: define a "phonologically valid" reward (no iltiqā'
as-sākinayn violations, no impossible consonant clusters, no
out-of-vocab haraqat combinations) and do GRPO on top of the supervised
model. Could push past 1.0% into territory neither SUKOUN nor Sadeed
reaches.

---

## Kimi K3 (arXiv:2607.24653, released July 16/27, 2026)

2.8T total / 104B active MoE. Native multimodal (vision). Frontier-level
on long-horizon coding and agentic tasks. Hybrid KDA-dominant
architecture: 3 KDA layers for every 1 full MLA layer.

### Component matrix

| Component | What it does | Take? |
|---|---|---|
| KDA (Kimi Delta Attention) | Linear attn w/ gated delta-rule + channel-wise gating | maybe — alt encoder for ablation |
| **AttnRes** (Attention Residuals) | Pass attention output across depth | **YES, drop-in** |
| Stable SMoE | MoE variant with stable load balancing | optional Tier 3 |
| LatentMoE | MoE with latent compression | no (scale) |
| **Per-Head Muon** | Per-head independent Muon optimization | **YES** (with MuonClip) |
| QK-Clip | Per-head logit clipping | **YES** (same as DeepSeek adoption) |
| NoPE | No positional embedding on some layers | combine w/ RoPE |
| 3:1 KDA:MLA hybrid | Long-context efficiency | no (short context) |

### Notes on the components

**KDA (Kimi Delta Attention):** Gated DeltaNet variant — a "linear
attention" mechanism that maintains a running delta-rule matrix as
memory, with finer-grained channel-wise gating than DeltaNet.

Could help char-level (diacritization is essentially "given recent chars,
what's the right haraqat?" — exactly what the delta-rule stores). But
our 256-512 context is short enough that full softmax attention works
fine. KDA's gain is asymptotic in sequence length. Verdict: try as an
alternative encoder for ablation only, not the primary shipping model.

**AttnRes (Attention Residuals):** Pass the attention layer's output
across depth, not just the residual stream. Improves information flow
across layers in deep models. At 12 layers our model is shallow, but
AttnRes is essentially free — a one-line skip connection from layer N's
attention to layer N+1's input.

**NoPE + RoPE hybrid:** Kimi K3 uses No Positional Embedding on some
layers and RoPE on others. Interesting idea — could combine RoPE on
attention layers (for relative position) with NoPE on FFN sub-blocks (no
position needed). Implementation-wise this is just "where do we apply
the rotation," which is already a config choice.

---

## Updated sprint plan additions

Folding into the existing 1-week sprint at zero wall-clock cost (no new
tasks, just expanded scope on existing ones):

- **Task #181 — Encoder modernization (Mon):** now includes RoPE + Flash
  Attention + **mHC** + **AttnRes** + max_len 512. Single-PR change to
  `src/rabba/model.py` block definition.
- **Task #182 — Pretraining (Tue):** now includes ELECTRA objective +
  **MuonClip** optimizer + **QK-Clip** + **Per-Head Muon**. Single-PR
  change to `src/rabba/pretrain.py` and a new `MuonOptimizer` class.

These drop-in changes compound with the rest of the sprint stack
(ELECTRA + multi-task heads + augmentation + trie decoder + Noisy
Student + ensemble distillation).

### Recompounded DER projection

Adding mHC (~−7%) + MuonClip (~−5% via better training) + AttnRes (~−3%)
on top of the original sprint stack:

Original target: ~0.9% DER (beats SUKOUN 1.11% and Sadeed 1.2% at 30×
smaller).

Updated target: **~0.75% DER** (beats them by a wider margin).

Stretch (if MoE Tier 3 lands): **~0.6% DER** — territory neither SUKOUN
nor Sadeed reaches.

### Compute budget impact

Muon's 2× pretrain efficiency actually **saves** compute time on the
sprint:
- Pretrain (was 6h A100, now 3h): saves 3 GPU-h
- Supervised (no change): 24 GPU-h (3 seeds × 8h parallel)
- Noisy Student: 8 GPU-h
- Distill: 6 GPU-h
- Export + benchmark: 2 GPU-h

**New total: ~43 GPU-h** (down from 48). Wall clock unchanged
(~5 days). Cost ~$145 (down from $165).

---

## What we explicitly do not take, and why

For posterity, here's the list of DeepSeek V4 + Kimi K3 techniques that
do **not** apply to a 50M-param char-level model with 256-512 token
context, so future contributors don't re-litigate them:

1. **MLA** — KV cache is tiny, low-rank projection adds complexity
2. **NSA / DeepSeek Sparse Attention** — sequence length too short
3. **MTP** — non-autoregressive task
4. **Engram** — covered by trie decoder at our scale
5. **3:1 KDA:MLA hybrid** — short context, doesn't pay back
6. **LatentMoE** — MoE with latent compression, only matters at
   multi-billion-param scale
7. **R1 RL** — supervised labels exist; v2 bet only

---

## References

- DeepSeek V4 technical report: `arXiv:2606.19348` (April 2026)
- mHC (Manifold-Constrained Hyper-Connections): `arXiv:2512.24880`
  (Dec 2025)
- Engram (Conditional Memory via Scalable Lookup): `arXiv:2601.07372`
  (Jan 2026)
- DeepSeek V3 technical report: `arXiv:2412.19437` (Dec 2024)
- DeepSeek V2 (MLA introduction): `arXiv:2405.04434` (May 2024)
- DeepSeek R1: `arXiv:2501.12948` (Jan 2025)
- NSA (Native Sparse Attention): `arXiv:2502.11089` (Feb 2025)
- Kimi K3: `arXiv:2607.24653` (July 2026)
- Kimi K2 (MuonClip): `arXiv:2507.20534` (July 2025)
- Kimi-Linear (KDA): `arXiv:2510.26692` (Oct 2025)
- Muon optimizer: Keller Jordan, github.com/KellerJordan/muon (2024)

---

## TL;DR

| Take | Skip |
|---|---|
| mHC | Engram, MLA, NSA, MTP, R1 RL |
| MuonClip + QK-Clip + Per-Head Muon | KDA:MLA 3:1 hybrid |
| AttnRes | LatentMoE |
| DeepSeekMoE aux-loss-free (if MoE) | |

Two drop-in changes (#181, #182) compound to ~−15% additional relative
DER. Sprint target drops from ~0.9% → **~0.75% DER**. Pretrain compute
saves 3 GPU-h thanks to Muon. Wall-clock unchanged.

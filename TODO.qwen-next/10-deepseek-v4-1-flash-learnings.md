# 10 — DeepSeek-V4.1-Flash learnings, mapped to our stack

Source: DeepSeek_V41_Tech_Report.pdf
(huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash, fetched 2026-09-10).
Model class: 552B multimodal MoE, 8B active prefill / 16B decode,
1M context. The report's subject is KV-cache compression at serving
scale — not our problem — but five of its components and two of its
recipe claims map onto levers we have open, and one unparks a parked
probe. Recorded so we don't re-derive it.

## 1. Speculative decoding — UNPARK 04 (the parked objection is void)

TODO 04 parked draft/verify because "the draft model would itself need
training + a parity story — not free." That objection expired: we now
ship the drafter. ara-diac-layerdrop-1.0-int4 (190M, 95 MB) and
ara-diac-small-2.1-int8 (300M) share the byte vocabulary, byte table,
and greedy decode; both are released, checksummed, and parity-gated.

DeepSeek's DSpark (§2.4.3) adds the design shape worth copying:
a small drafter emits K tokens in one pass, the verifier checks all K
in one pass, and a confidence estimate schedules K adaptively. With
greedy verification the verifier's argmax is authoritative, so outputs
are byte-identical to non-speculative greedy — no parity story needed,
just a speed one.

- Honest scope: this speeds the 2.1 tier. The lite tier stays as the
  standalone fast path; speculative decode is a middle tier that only
  pays when users want 2.1 quality at lower latency.
- Step 1 RESULT (2026-09-11, complete, 25/25 golden rows):
  mean acceptance **0.9886** (min 0.955, median 0.991),
  **8.85 tokens per verifier pass** at K=8 (incl. bonus tokens).
  Exactness: all 18 rows checked against the verifier's plain-path
  greedy are byte-identical (spec_loop == verifier greedy, the
  output-preservation theorem holds on real quantized artifacts);
  plain==KV on every checked row. Gate (>=0.9 acceptance) cleared
  decisively. Runtime implementation: interscript-ts PR #77 (merged)
  - SpeculativeModel exported from interscript/ml, tiny-fixture +
  real-pair e2e green. Remaining: playground/API tier exposure.


## 2. Multi-teacher domain-routed distillation — cheap untested lever

§5.2.4: the final OPD stage trains from 40+ architecturally
heterogeneous teachers, selecting the best teacher *per domain* ("the
best teacher for each domain may come from a different stage of model
development"). This is standard practice for them, not research.

Our residual program closed corpus volume, domain swap, and on-policy
GKD — but never a teacher *mixture*. r6 and r7 are both measured
full-set (2.5793 / 2.29), and frontier-predictions-v1 already ships
both teachers' raw predictions.

- Step 1 result (2026-09-11, measured): r6 was run over the same
  1200 rows under the windowed protocol (Modal probe, preds at
  /checkpoints/probes/ on rababa-checkpoints) and sliced by the
  benchmark's filename domains. r7 wins EVERY domain — classical a
  tie (r6 1.37 / r7 1.36), news r7 (3.52/3.21), wiki r7 strongly
  (3.29/2.08). r6 full-set re-scores 2.5997 vs the published 2.5793
  (0.02pp protocol-window variance, same model). The routing
  hypothesis is NEGATIVE for the available teacher pair: no domain
  exists where routing to r6 helps. Combined with the student-side
  slice (gap concentrated on news/wiki, NOT classical), the residual
  is neither classical coverage nor teacher selection — the remaining
  category is student-side: capacity/interaction (TODO.impl/04
  trained-lite, /10 lexical memory) or runtime (TODO.impl/01).

## 3. Head-wise Muon — external validation, wire-in already pending

§2.5: Q/K weights split per head before the Muon update, "outperforms
vanilla Muon," independently validated in GLM 5 and Kimi-K3. We built
the module in v0.5.0; the wire-in is still pending acceptance criteria
(TODO.arabic/18). Three-lab external validation upgrades this from
"nice to have" to "default for the next teacher run." No new code —
flip the config flag, run the spec.

## 4. Layer compression must be trained, not pruned — reframes the Hebrew collapse

CSA2's Reuse-mode layers (§2.3.1) share KV computed by other layers —
functionally a depth cut — but the mode assignment is static *during
training*, so the network adapts around what it will and won't
compute. CORRECTION (2026-09-11, spec audit): our lite rung is NOT an
untrained copy — run-009-layerdrop-6ep is a full 6-epoch sequence-KD
distill whose INIT is a verbatim kept-layer copy from byt5-small.
"Verbatim layer copy" describes the init bridge, not absent training.
What is actually untested in the lite cell: the INIT SOURCE. run-009
drops layers from generic pretrained byt5-small; DeepSeek's pattern
(compress from TRAINED weights, then adapt) suggests dropping from the
trained 2.1 student instead.

- Candidate: lite-2.0 = layer-drop init from
  run-007-r7-muon-6ep/best (the shipped 2.1), then the identical
  6-epoch distill. Single variable vs run-009's 5.78: init source.
- The Hebrew collapse (77.48) stands as a cross-lingual depth
  replication failure with a known recipe confound (logit-KD 3ep vs
  Arabic sequence-KD 6ep), not as evidence about trained vs untrained
  compression.

## 5. KV-cache int8 in the runtime decoder — one runtime lever

§2.4.4's precision hierarchy: main KV → FP4 (QAT), SWA KV → FP8
("sensitivity"), head fp32. Our head-fp32 discipline already matches
their reasoning. Unexplored on our side: the runtime decoder's KV
cache is fp32 today; caching K/V at int8 with per-head scales would
halve decode memory bandwidth in the browser, where bandwidth is the
binding cost. Gate with the existing quantized contract (cer_delta +
decode-health smoke), not byte parity — same class of claim as
weights-quantization, already scoped by golden-v1.

## 6. Smaller items

- **Sinkhorn-balanced update for embedding + head (Alg. 1)**: replaces
  Adam with momentum + Sinkhorn row/col normalization, momentum-only
  buffer, "empirically outperforming Adam." K=11, τ=1e-3, γ=0.18. Our
  byte-vocab embedding and head are tiny, so the memory saving is
  irrelevant; the quality claim is untested at our scale. Our
  log-domain Sinkhorn code (mHC work) transfers. Candidate
  single-variable run, low priority.
- **mHC single-pass shift (§2.4.1)**: consume the previous block's
  mixing coefficients (A_{l-1}) to kill the kernel dependency,
  halving activation traffic; "negligible degradation." Adopt if we
  ever resume n-stream mHC students (TODO.arabic/19).
- **Engram is a hash lookup, not retrieval (§2.4.2)**: TODO.arabic/10
  and TODO.hebrew/11 describe Engram as retrieving similar past
  *training examples*. The actual mechanism is n-gram-hash-addressed
  embedding tables summed into the forward pass (orders {2,3,4},
  8 hash heads, 2048-dim per order, distinct prime table sizes,
  FP8 storage, modules at layers 1 and 14, 5× LR, tables updated with
  the Sinkhorn-balanced rule). 196B of their 748B parameters live
  there — memorization decoupled from compute. For diacritization the
  lexical-idiom story is real (haraqat are n-gram-driven), but browser
  size caps the table (int8 4M-entry × 64-dim ≈ 64 MB is the ceiling).
  Architecture-level lever; only worth it if §2's teacher-mixture also
  fails, since it changes export and runtime.
- **Diacritization-depth conditioning**: the reasoning-effort scalar
  (§5.1.4, effort 1–100 → low/high/max API tiers) suggests training a
  depth control (none/light/full tashkeel) as an input-side signal —
  one model, one knob, interpolated at inference. Product feature,
  needs a new student run; user decision, registered for the roadmap.

## 7. External validation for decisions already made

- §4.1: model-generated content is treated as "implicit duplication"
  and filtered from pretraining — machine-labeled data demoted, not
  promoted. Validates the knesset-v6 (Dicta labels) weak-pretrain-only
  decision for Hebrew.
- §5.1: "the marginal return of engineering the data and environment
  pipeline substantially exceeds that of algorithmic novelty in
  post-training." Consistent with our closed program: corpus and data
  levers measured; what remains is interaction-level, which is why
  §2 (teacher mixture) is the probe that fits.

## 8. Not applicable, with reasons

| Report component | Why not |
|---|---|
| CED, CSA2, cross-layer KV reuse | Long-context serving scale; our windows are ≤1400 B, KV storage is not a cost |
| SWA Bounded Replay, persistent KV mgmt, EPD disaggregation | LLM serving infra; no prefix-reuse workload |
| Hierarchical Sparse Indexer | Reduces indexer cost at 1M context; nothing to index at ours |
| FP4/MXFP4 formats | Hardware-GEMM formats; our targets are CPU/web ONNX (int8/int4 weights) |
| 1M-context pretraining, 45T tokens | Avoiding that scale is the point of the client tier |
| Agent task synthesis, DSec, multi-agent, effort-RL machinery | No RL anywhere in our pipeline; RL-for-diacritization is closed negative (knowledge-limited at SFT convergence) |
| Async OPD infrastructure | Our distillation is offline sequence/logit KD; no rollout phase |

## Status

- [ ] §1 step 1: acceptance probe on golden-v1 (CPU, no training)
- [ ] §2 step 1: per-domain teacher slice from released predictions
- [ ] §3: head-wise Muon wire-in spec executed on next teacher run
- [ ] §4-§6: registered as candidates, costed, awaiting user prioritization

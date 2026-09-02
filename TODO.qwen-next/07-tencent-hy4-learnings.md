# 07 — Tencent Hy4-preview learnings, mapped to our stack

Source: huggingface.co/tencent/Hy4-preview model card (fetched
2026-09-01; released days earlier — 770B total / 49B active, 78
layers, 1M context). Companion: Hy4-preview-FP8. The card is candid
that it is a preview and discloses little pre-training recipe — the
substance is architectural + serving-level, with one directly
actionable training direction for us.

## 1. iHC (identity Hyper-Connections, 4 residual streams) — ACT ON THIS

Hy4 routes inter-layer information through 4 identity-based residual
streams. GLM-5.3-Flash ships the mHC variant (TODO 06). Two frontier
labs independently adopting hyper-connections is a signal, and it
lands exactly on our open frontier rung: the stitch-down from
ByT5-small pretraining (ridge-fit width bridge, per
PUBLICATION-NOTES). Hyper-connections were designed for the problem
our ridge bridge solves ad hoc — preserving trainable identity paths
across width surgery. Candidate E-item: width-stitch with a
hyper-connection reparameterization vs the ridge bridge, same data
and budget as the existing frontier rungs (controlled, comparable to
the layerdrop rung). This is a paper-B experiment, not a client-tier
change.

## 2. Native MTP head (10B/0.7B active, 3 speculative tokens) — UN-PARK TRIGGER?

Hy4 ships multi-token prediction natively for speculative decoding.
Our TODO 04 (speculative decoding probe) is parked on the condition
"API latency data shows p95 decode binding". Check the IMF runtime
benchmarks (RESULTS.md, E1 node tier): if greedy KV decode dominates
wall time for the client tier, an MTP-style extra head on the student
is the cheap version of this — diacritization output is locally
byte-predictable (input letter + haraqat pattern), so a 2-3x step
reduction is plausible at ~0.2% size cost. Decision needed: pull the
benchmark numbers and either open the E-item or re-park with the
measured latency as the recorded reason.

## 3. Gated DSA + IndexCache (arXiv 2603.12201) — paper note only

Sparse attention with cross-layer index reuse; same family as
GLM-5.3-Flash's hybrid sparse+linear. Our windows are <=1400B —
attention is not the bottleneck, and quadratic attention staying
cheap at 300-580M is part of the client-tier story. One related-work
line: IndexCache's "compute routing once, reuse across layers" is the
serving-side cousin of what our KV cache does for decode state.

## 4. What the card does NOT give us

No pre-training data/curriculum, optimizer, RL algorithm, load
balancing, or long-context recipe — "we scaled model size, context,
and data" plus "a substantially larger post-training run". The only
post-training substance: expert-built task data per domain (SWE,
office, game-dev, science) co-designed with products (CodeBuddy,
WorkBuddy). That is the pattern our r7 news-domain adaptation
followed at our scale — a validation, not a new method.

## 5. Evaluation methodology contrast (paper-useful)

Their headline comparison is 163 internal experts, 203 tasks, blind
side-by-side vs GLM 5.3 / Kimi K3 (win/tie/loss) — strong but
non-reproducible. Our external rows are public-protocol,
reproducible-by-construction (the same benchmark+evaluator+decode
disclosure that caught the GLM-5.3-Flash orthography regression). One
sentence for the paper's evaluation section: internal blind pairwise
and public-protocol benchmarks are complementary; only the latter
adversarially constrains vendor-reported gains.

## 6. reasoning_effort defaults "high", opt-out via chat_template_kwargs no_think

Third vendor with a non-obvious reasoning knob (GLM-5.3-Flash:
absent/invalid -> MAX, thinking unrejectable; Hy4: default high,
no_think via template kwargs). Standing rule reinforced: every LLM
row must disclose the full decode protocol, and the response-side
reasoning_content tripwire stays. If we ever add Hy4 to the SadeedDiac
LLM rows: read the opt-out semantics BEFORE trusting any "plain"
run.

## Not applicable (recorded for completeness)

- 770B/49B active MoE, 256 experts top-8 + shared — more evidence for
  the parameters-!=compute axis, no action for us.
- 1M context training recipe — undisclosed; our windows are bounded.
- FP8 companion artifact — our fp16/int8 policy is set by E1.

## Cross-reference (2026-09-01)

07-hy4-learnings.md is the canonical Hy4 verdict — its MTP framing
(as a TRAINING auxiliary, not serving-side speculation; decode is
measured non-binding at our sizes per benches E1/E2) overrides the
serving framing in my section 2, and its domain-data prioritization
feeds the 08 plan. My unique content: the evaluation-methodology
contrast and the reasoning_effort knob survey.

# 09 — Claude Fable 5.1 / Mythos 5.1 learnings, mapped to our stack

Sources: the announcement (anthropic.com/claude-fable-and-mythos-5-1,
fetched 2026-09-01) and the news index (anthropic.com/news). Verified
after the initial web search returned only SEO/rumor noise with
contradictory claims — the guessed `/news/claude-fable-5-1` URL 404'd;
the real page is the combined Fable-and-Mythos announcement dated
2026-09-01. Scope: same as 06/07 — what applies to interscript-ml /
rababa, what doesn't, and why.

## 1. Reasoning-effort defaults are per-surface (fourth data point)

"Fable 5.1 defaults to High effort in Claude Code, and to Medium in
Claude Cowork and on Claude.ai." The default for the *same model*
varies by product surface, because the vendor is tuning
cost/latency/quality per context.

- Extends the 06/07 survey: GLM (absent/invalid → MAX), Hy4 (default
  high, `no_think` opt-out), Anthropic (per-surface defaults).
- **Rule reinforced**: never rely on a provider default at a call
  site — it is not even stable *within* one vendor's model across
  their own surfaces. Our `eval_sadeed_glm.py` pattern (explicit
  knob, refusal to start without one on glm-5.3*, effort in the
  checkpoint filename + protocol line, `reasoning_content` tripwire)
  is the right shape for every future API eval.

## 2. Applicability map, item by item

| Fable 5.1 item | Verdict for us | Why |
|---|---|---|
| per-surface effort defaults | **adopted** (§1) | 4th data point for the knob survey; validates explicit-knob rule |
| reward-hacking audits via "natural language autoencoders on internal thinking" | paper-note only | interpretability on *their* reasoning traces; our byte-level seq2seq students have no thinking traces. Worth one line in the paper's interpretability/future-work context, nothing actionable now |
| strengthened distillation defenses (thinking-transcript context-editing restricted) | no impact | we distill our *own* teacher (rababa r6/r7 → students), never a hosted frontier model — and LLM-as-teacher for diacritization is banned outright (hallucinated haraqat) |
| invisible text watermarking + private-preview detection API | no impact | we ship weights as sha-pinned artifacts, not LLM text; our corpora are classical/protected texts, not model output. If we ever ingest web-scale LLM-generated Arabic/Hebrew text into training, watermark taint becomes a provenance question — current hygiene (decontamination scans, dedup) is the mitigation |
| cache reads cut to $0.25/M (25–45% workload savings) | technique noted | provider-specific (Anthropic); our GLM evals run on z.ai. General lesson applies if we ever host on Anthropic: put the static instruction/prompt prefix first and identical across requests so it caches — our sweep template already has that shape |
| safety-FP reductions (bio 85% fewer benign fires, cyber 60% fewer FPs) | n/a | eval-convenience for agentic coding, not a training technique |
| agentic benchmarks at the top (Terminal-Bench-Science 52.6, Terminal-Bench 4.0 55.8, GDPval-AA 1853; 2.5x GPU-kernel speedups, ~50% protein-binder hit rate) | **thesis support** | frontier effort targets agentic/science work, not classical-knowledge tasks — same signal as the GLM-5.3 regression we measured (wrong-haraqat axis up vs 5.2). Dedicated distilled models remain the right call for diacritization |
| protein-binder / GPU-kernel agentic optimization | n/a | no analog in diacritization; no reward-verifiable iteration loop for haraqat (RL already ruled out — knowledge-limited, not exploration-limited) |

## 3. What is NOT disclosed (and why that matters)

No architecture, no RL algorithm, no training-data or pretraining-
compute details, no synthetic-data or agent-training method, no
curriculum, no effort-level implementation, no watermark algorithm.
Same wall as 06 (GLM) and 07 (Hy4): frontier labs publish capability
and safety framing, not mechanics. Nothing here is borrowable for our
distillation rungs — the levers stay on our side: data scale/register
diversity (E6), supervision quality, GKD — not architecture copying.

## 4. Standing conclusions (delta over 06/07)

- The knob survey is complete enough to freeze: **every** future LLM
  evaluation must state the explicit effort/thinking setting next to
  its numbers; provider defaults are per-surface and unstable.
- Frontier-vs-dedicated thesis now has three corroborations: our
  GLM-5.3 measurement (9.90 raw, worse than 5.2 on the wrong-haraqat
  axis), GLM/Hy4 release framing, and Fable 5.1's benchmark choices.

# 07 — Tencent Hy4-preview learnings, mapped to our stack

Sources: model card (huggingface.co/tencent/Hy4-preview, fetched
2026-09-01), hy.tencent.ai/research/hy4-preview, vLLM recipes page.
No arXiv technical report exists yet — unlike the GLM-5 paper, there
are NO disclosed training specifics (no optimizer, RL, curriculum,
token budget, or MoE training tricks). What the card carries is
architecture + a post-training data philosophy. Scope: what transfers
to byte-level seq2seq at our scale (<=580M teachers, <=300M
students), what does not, and why.

## 1. Architecture inventory (for the papers' related-work)

- MoE 770B total / 49B active; 78 layers (first dense, rest MoE);
  256 routed + 1 shared expert, top-8 routing
- Gated DeepSeek Sparse Attention + IndexCache (cross-layer sparse
  index reuse); 64 heads, Q compressed to 2048 / KV to 512
- iHC: identity hyper-connections, 4 residual streams
- MTP layer: 10B/0.7B for speculative decoding; 1M context

## 2. Applicability map

| Hy4 idea | Verdict for us | Why |
|---|---|---|
| MTP as auxiliary training objective | **candidate rung (E5?)** | per-position multi-step heads densify supervision for decode-bound byte students; might harden against the repetition pathologies. Cheap probe: MTP-aux head on ByT5-small + Muon, same labels/gate |
| expert co-created data ("built around the work they ship") | **adopted principle** | our ~2.0pp domain residual is exactly the axis this addresses -> bumps the Tashkeela++/label-scale rung (rababa PR #1, open) |
| iHC (4 residual streams) | paper note only | adjacent to the microkimi stitch/geometry observations; on frozen ByT5 students it is matrix surgery, and the width law says the pretrained geometry is load-bearing |
| Gated DSA / sparse attention | not applicable | our windows are <=1400B; attention is not the serving bottleneck (measured, bench E1/E2) |
| MoE student | closed territory | E2's PKM probe: sparse capacity below the pre-registered bar; gap is optimization + domain, not capacity |
| "ship early, hear what breaks" | already our practice | previews/runs |

## 3. What we deliberately do not adopt

- Frontier-scale MoE anything (the client tier's point is avoiding it)
- LLM-as-teacher (standing rule; unchanged by Hy4's data philosophy)
- Speculative decoding in IMF decode (greedy KV is already fast at
  our sizes; MTP's value here would be as a TRAINING aux, not serving)

## 4. Net

One cheap testable idea (MTP-aux distillation rung), one principle
reinforcement (domain data rung rises in priority), zero disclosed
training mechanics to borrow. Revisit if a technical report lands.

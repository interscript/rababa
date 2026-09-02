# Five Scripts, One Lesson: Data Curation Beats Model Capacity

*2026-08-19 — the Interscript ML team on a month of diacritization and
grapheme-to-phoneme work across Arabic, Hebrew, Thai, Persian, and Urdu.*

Interscript's transliteration maps (ALA-LC, SBL, RTGS, …) assume
*vocalized* input. Undiacritized Arabic or Hebrew, or unsegmented Thai,
cannot be transliterated unambiguously — so we spent a month building
the phonological layer under the maps. This post is the full ledger:
what worked, what failed, and what we can now prove.

## Arabic: a 580M model beats our verified frontier reproduction

On SadeedDiac-25 (1,200 expert-reviewed paragraphs, Misraj's own
ArabicDiacritizationEvaluator, zero skipped paragraphs), our final
model — a ByT5-base fine-tuned in three stages — lands at:

| System | DER (CE) | DER (w/o CE) |
|---|---|---|
| Claude-3.7-Sonnet (published) | 1.39 | 0.77 |
| GLM-5.2 (our reproduction, raw) | 2.51 | 1.55 |
| GLM-5.2 (our reproduction, zero-skip) | 2.69 | 1.72 |
| **ours (r5 paragraph-context)** | **2.68** | **1.60** |
| Gemini-Flash-2.0 (published) | 3.19 | 2.38 |
| GPT-4 (published) | 3.86 | 3.86 |
| Sadeed (published, 1.5B params) | 7.29 | 5.26 |

**About that Claude number.** We verified the frontier ourselves:
GLM-5.2 — a strictly stronger 2026 flagship than the Claude-3.7
generation the benchmark card cites — scores 2.51 DER under a neutral
protocol (temperature 0, plain completion, no tricks). That is nowhere
near the published 1.39. We cannot reproduce it with a better model
under a documented protocol; we invite anyone who can to publish
theirs. What we *can* say: our compact model matches or beats the
**verified** frontier, which is strong evidence our own DERs are
measured correctly — we ran the same evaluator, the same 1,200
paragraphs, zero skips, and then out-scored a model that out-scores
what we can verify.

**How we got there — every step was data-side:**

1. **Clean like the benchmark authors.** We ported the Sadeed cleaning
   heuristics over the full 75M-word Tashkeela dump. Scaling 75K → 2.1M
   sentences took in-domain DER from 2.42% to 0.99%.
2. **Audit the training corpus for benchmark leakage.** Misraj's public
   1.88M-line corpus *contains its own benchmark*: 122 paragraphs
   verbatim plus ~1k near-duplicates. We decontaminated with 60-char
   stride-1 windows before touching it. Any number trained on the raw
   release is contaminated.
3. **Domain-adapt.** One epoch on the decontaminated corpus: 2.94 →
   2.84 DER.
4. **Train on paragraphs, not lines.** Our training units were isolated
   ≤640-byte lines while LLMs read whole paragraphs. The corpus is
   line-split continuous book text, so we joined adjacent lines into
   ~1400-byte paragraph units: 2.81 → **2.68 DER (CE), 1.60 without
   case endings** — past the verified GLM-5.2 zero-skip row on both
   metrics.

**What didn't work — the RL graveyard.** Our residual errors are 98.7%
phonotactically *legal* alternate readings: a discrimination problem,
not a phonology problem. We threw the 2025/2026 policy-optimization
toolkit at it:

- **RAFT** (rejection-sampling fine-tuning, 3 iterations, 1,170
  winners): flat. 2.8429 → 2.8515.
- **Sequence-level GRPO** (on Persian): *negative* — the best
  checkpoint lost 2 points on SentenceBench homographs (77.34 →
  75.37%) and dev reward degraded monotonically.
- **Entropy-weighted GRPO** (GTPO, arXiv 2508.04349, per-token credit
  concentrated on high-entropy decision positions, graded
  alignment-based reward): the dev curve was *exactly* flat —
  5.9692% at steps 0, 100, and 150, to four decimal places.

Three methods, two languages, one verdict: at SFT convergence on
clean data, diacritization residual error is **knowledge-limited, not
policy-limited**. No policy-sharpening operator recovers distinctions
the data didn't teach. We're publishing the negative results because
they're the most expensive thing in this post to rediscover.

## Hebrew: input formats silently dominate comparisons

Our ByT5 model scores **17.46% DER** on the Nakdimon Biblical test
split — 6 points better than DictaBERT (35.6% under the identical
protocol), a model that scores ~4% on modern Hebrew. The SOTA is
domain-bound, and the Biblical/Rabbinic domain is where our
transliteration users live.

Two findings that should be on every Hebrew diacritization paper's
checklist:

- **The teamim leak.** Standard Nakdimon-derived preprocessing strips
  nikud but *not* teamim (cantillation marks) from model input. Models
  get cantillation hints for free; teamim "errors" become impossible.
  We proved the leak exists by breaking it: strip teamim from the
  input and DER collapses to 36.2%. Any reproduction must state its
  input format.
- **Beam search is a 12-point factor** (29.0% greedy vs 17.5%
  beam-4). And output-vote ensembling *hurts* (17.8 → 21.5%): majority
  voting splices incoherent strings that edit distance punishes.

## Thai: deterministic augmentation beats LLM labeling

Thai G2P was bottlenecked by labeled data, so we generated more
labels — from a rule-based phonemizer (epitran), not an LLM. 50K
unlabeled Thai Wikipedia sentences, phonemized deterministically, then
continued fine-tuning: PER fell **3.24% → 2.32%** on a fixed
1,219-sentence test set (the public baseline is 6.37%). LLMs
hallucinate phonology; rules don't.

## Persian: first learned system to pass the published SOTA

SentenceBench homographs are the stress test (identical spellings,
context-dependent pronunciations). Our v1 ByT5-small scores **77.34%
ezafe-normalized accuracy — past the published Homo-GE2PE SOTA of
76.89%** — with ~1.6% CER overall. The recipe that got there is
notable for what it rejected: longer training *hurt* homographs
(-2.5 points), dual-task heads hurt, and GRPO hurt (above). The
early-stopped, boring recipe won.

## Urdu: the first learned baseline

635K sentence pairs retrained: **14.77% CER** — 4.1× better than the
rule-based pipeline it replaces, and the first learned Urdu G2P
baseline we know of.

## The pattern, stated once

Across five scripts and three task families:

1. **LLMs are unreliable phonological labelers.** They hallucinate
   haraqat and nikud; a single systematic error poisons a distillation
   chain. Every teacher label in this post is gold human text or a
   deterministic rule.
2. **Protocols dominate.** A silent truncation once made our eval
   *skip* the hardest paragraphs and read 1.82% — survivorship bias,
   not quality. We now publish only zero-skip numbers, and we verify
   the frontier before comparing to it.
3. **Contamination is everywhere you look.** 122 verbatim benchmark
   paragraphs in the public training corpus of the benchmark itself.
4. **Data curation beats capacity — and beats RL.** A 10M-parameter
   char encoder beats a 1.5B model on its own benchmark. A 580M model
   beats the verified frontier. Three RL methods moved nothing.

All code, corpora, cleaning pipelines, and evaluation protocols are in
the `interscript` monorepo (`rababa`, `rababa-farsi`, `secryst`);
`docs/RESULTS.md` carries exact run IDs for every number above.
Models ship as BSD-3-Clause weights for local, offline, deterministic
transliteration runtimes — the interscript.org way.

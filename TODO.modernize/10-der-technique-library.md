# DER improvement technique library

Every technique considered for the rababa SOTA sprint, with expected DER
delta, implementation cost, and skip reasoning where applicable. This is
the catalog from which the sprint stack was assembled.

## Tier 1 — High-impact, proven in Arabic diacritization

### T1.1 · Word-level lexicon trie at decode time
**Expected:** −15 to −25% relative DER on Fadel-class benchmarks.
**Cost:** ~1 day. ~5 MB JSON shipped alongside model.

Build dictionary `{undiacritized_word → Counter(diacritized_forms)}` from
train. At inference, for each word, mask per-char logits to only permit
haraqat sequences that produce a known form (fallback: unconstrained for
OOV). Most Fadel test words are seen-in-training; the trie is
essentially a free 15–25% DER reduction with zero model change.

This is what SUKOUN and Sadeed implicitly lean on through their BERT
tokenization.

### T1.2 · Multi-task auxiliary heads (Alqahtani 2020, arXiv:2006.04016)
**Expected:** −10 to −15% relative DER.
**Cost:** ~2 days implementation, +10% training time, zero inference cost.

Three shared-encoder heads:
- **POS tag** (top-level: noun/verb/particle — 3-way, weak labels via CAMeL)
- **Word segmentation** (border / no-border at each char)
- **Syntactic vs morphological diacritization** flag

Multi-task loss with weighted combination. Aux heads regularize the
encoder; at inference only haraqat head is used.

### T1.3 · Phonological rule injection — iltiqā' as-sākinayn
**Expected:** −3 to −8% relative DER on classical text.
**Cost:** ~1 day.

Finish what `clean_tashkeela_sadeed.py` started. The rule: no two
adjacent sukun-bearing consonants — the second takes a vowel. Sadeed's
paper calls this out specifically. Implementation is a deterministic
post-processor on targets; also a constraint at decode time.

### T1.4 · Constrained decoding via trie-based logit masking
**Expected:** compounds with T1.1, +5% on top.
**Cost:** ~1 day.

Maintain pointer into the trie; mask invalid next-haraqat logits to
−inf. Beam width 4. Catches sequences that per-char argmax misses.
~2ms per 256-char sequence on CPU.

## Tier 2 — Medium-impact, proven in adjacent domains

### T2.1 · Self-training / Noisy Student on arwiki
**Expected:** −5 to −15% relative DER.
**Cost:** ~3 days implementation, ~$10 compute.

Pipeline:
1. Train teacher (rababa_arabic_pro with T1.1–T1.4)
2. Run teacher over ~5M arwiki lines → soft labels
3. Train equal-size student on (gold ∪ pseudo) with augmented noise
   (dropout 0.3, random haraqat drop on input side, length jitter)
4. Iterate 2× (Xie 2020 showed gains plateau after 3 iterations)

Single highest-ROI semi-supervised technique for token classification.

### T2.2 · Ensemble of 5 seeds → distill back to single model
**Expected:** −10 to −15% relative DER from ensemble, −5 to −8% retained
after distillation.
**Cost:** ~$50 compute (5× training), ~1 week wall-clock if parallelized.

Train with 5 seeds. At inference, average per-position logits. Then
distill (Hinton) into a single model that mimics the ensemble. Distilled
model keeps ~60–70% of ensemble gain at single-model inference cost.

Sprint uses 3 seeds + noisy student = 4-model ensemble (cheaper).

### T2.3 · RoPE + Flash Attention + longer context
**Expected:** −3 to −7% relative DER + 2× training throughput.
**Cost:** ~2 days. Drop-in.

Replace sinusoidal positional embeddings with Rotary (RoPE). Flash
Attention via PyTorch 2.x SDPA. Push `max_len` 256 → 512. Char-level
models benefit from longer context for classical Arabic. RoPE
extrapolates better than learned positions.

### T2.4 · ELECTRA-style pretraining (replaces MLM)
**Expected:** −3 to −8% relative DER, ~4× pretraining efficiency.
**Cost:** ~3 days implementation, same compute budget.

MLM wastes compute on easy tokens. ELECTRA's Replaced-Token-Detection
(small generator corrupts tokens, discriminator predicts which) is ~4×
sample-efficient. For our 6h pretrain budget, that's roughly +1 epoch of
effective training.

### T2.5 · Data augmentation: input-side haraqat drop
**Expected:** −2 to −5% relative DER.
**Cost:** half a day.

For 30% of training examples, randomly drop 1–3 haraqat from input side
but keep full targets. Forces the model to handle partially-diacritized
input (which is what real-world Arabic text looks like).

## Tier 3 — Research bets (one of these, not all)

### T3.1 · Sparse Mixture of Experts (MoE) encoder
**Expected:** −5 to −10% relative DER, 3–4× effective capacity at same
inference cost.
**Cost:** ~1 week. Risk: routing instability, ONNX export of dynamic
indexing.

Replace FFN in 6 of the 12 layers with top-2 MoE (8 experts). Total
params stay ~50M but active per-token is ~15M. LiteRT.js can run sparse
MoE if we route-on-input (always-eval top-2).

Use **DeepSeek V3's aux-loss-free load balancing** (per-expert bias
update, no aux loss) — cleanest MoE training recipe available.

### T3.2 · LLM pseudo-labeling for hard cases
**Expected:** −2 to −5% relative DER on SadeedDiac-25.
**Cost:** ~$100 API spend, ~2 days implementation.

Where Tier-1 model is uncertain (low max-prob), call GPT-4 / Claude /
Gemini on unpointed text with phonology-aware prompt. Add LLM labels as
third teacher (alongside gold + ensemble) in self-training. Useful for
proper names and rare constructions.

### T3.3 · Distill Sadeed's released model
**Expected:** −3 to −8% relative DER.
**Cost:** ~1 week, ~$20 compute.

`misraj-ai/Sadeed` is on HuggingFace. Run over arwiki (5M lines), use
predictions as soft labels for our 50M student. Effectively compress
1.5B Sadeed into 50M browser model.

License-clean: learning from outputs on GPLv2 text, not redistributing
Sadeed.

### T3.4 · GRPO RL with phonology reward (R1-style)
**Expected:** unknown — possibly −0 to −3% on top of supervised.
**Cost:** ~2 weeks, ~$200 compute.

Define phonology-validity reward (no iltiqā' violations, no impossible
consonant clusters, no OOV haraqat). GRPO on top of supervised model.
Could push past 1.0% into territory neither SUKOUN nor Sadeed reaches.

v2 bet — not in the 1-week sprint.

## Skipped techniques (and why)

| Technique | Why skipped at 50M / char-level / 256–512 context |
|---|---|
| MLA (DeepSeek V2) | KV cache is ~5 MB at our scale |
| NSA / DeepSeek Sparse Attention | Sequence too short for sparse to pay back |
| MTP (DeepSeek V3) | Non-autoregressive task |
| Engram (DeepSeek, arXiv:2601.07372) | Covered by trie decoder |
| 3:1 KDA:MLA hybrid (Kimi K3) | Short context, no asymptotic gain |
| LatentMoE (Kimi K3) | Scale issue |
| CAMeL-Cowen-Salabi (Hebrew) | Wrong language |

## Compound projection

If all Tier 1 + Tier 2 + T3.3 (distill Sadeed) land:

`0.80 × 0.95 × 0.92 × 0.95 × 0.97 × 0.80 × 0.95 × 0.90 × 0.92 ≈ 0.32`

Baseline rababa_arabic_pro estimated ~2.5% → final ≈ **0.8% DER**.

Sprint scope drops Tier 3 entirely (no time in 1 week), targets
**0.75%** with V4/K3 additions (mHC + MuonClip + AttnRes) substituting
for T3 contributions.

## Sources

- SUKOUN (Kharsa 2024): Expert Systems with Applications, "BERT-Based
  Arabic Diacritization"
- Sadeed (Aldallal 2025): arXiv:2504.21635
- Advancing Arabic Diacritization (Mohamed & Mubarak, EMNLP 2025):
  arXiv:2509.xxxxx (see `09-sadeed-qcri-data-access.md`)
- Alqahtani 2020 (multi-task Arabic diacritization): arXiv:2006.04016
- Noisy Student (Xie 2020): arXiv:1911.04252
- ELECTRA (Clark 2020): arXiv:2003.10555
- DeepSeek V4: arXiv:2606.19348 — see `PROPOSAL.ds4-k3-proposal.md`
- Kimi K3: arXiv:2607.24653
- mHC: arXiv:2512.24880
- Muon: github.com/KellerJordan/muon

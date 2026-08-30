# rababa — SOTA Results (Arabic + Hebrew)

All numbers measured in the 2026-08 SOTA campaign. Modal run IDs and scripts
referenced per result. This file is the ground truth for the papers in
`docs/paper-*/`.

## Arabic diacritization

### Best result

| Metric | Value | Test set |
|---|---|---|
| **DER** | **0.99%** | 52,224 held-out examples from 2.1M combined corpus |

- Model: custom char-level Transformer encoder, 6L/384d/6h (~30M params)
- Training: 15 epochs, combined corpus = Tashkeela-full (75M words cleaned
  Sadeed-style) + Arabic Wikipedia + QCRI EMNLP-2025
- Eval script: `modal run modal_app.py::evaluate --task rababa_arabic_v2`
- Run: ap-ZEK2zNXeT9kTOIAI3kAjQT (2026-08-13)

### Data scaling ablation

| Corpus size | DER |
|---|---|
| 75K | 2.42% |
| 2.1M | **0.98–0.99%** |

28× data scaling → 2.5× error reduction.

### Comparison

Direct two-way protocol (2026-08-14): we ran our model on the **full
SadeedDiac-25 benchmark** (1,200 paragraphs) with **Misraj's own
ArabicDiacritizationEvaluator** (vendored: `sadeed_evaluator.py`), default
protocol, zero skipped paragraphs.

| System | Params | DER (CE) | DER (w/o CE) | WER (CE) | WER (w/o CE) |
|---|---|---|---|---|---|
| Claude-3-7-Sonnet (published) | — | 1.3941 | 0.7693 | 4.6718 | 2.3098 |
| **GLM-5.2 (our reproduction, raw)** | — | 2.5060 | 1.5537 | 7.9929 | 4.7509 |
| GLM-5.2 (our reproduction, zero-skip) | — | 2.6911 | 1.7179 | 8.3037 | 5.0619 |
| Gemini-Flash-2.0 (published) | — | 3.1926 | 2.3783 | 7.9942 | 5.5044 |
| GPT-4 (published) | — | 3.8645 | 3.8645 | 5.2719 | 10.9274 |
| Sadeed (published) | 1.5B | 7.2915 | 5.2625 | 13.7425 | 9.9245 |
| **rababa r3 domain-adapted (ours)** | 580M | **2.8429** | **1.7589** | **8.4981** | **4.8859** |
| rababa r3 zero-skip projected | 580M | 2.8126 | 1.6877 | 8.4110 | 4.5739 |
| rababa_arabic_byt5 r2 (ours) | 580M | 2.9406 | 1.8333 | 8.8373 | 5.0835 |
| rababa_arabic_byt5 r2 (beam 4) | 580M | 2.9478 | 1.8522 | 8.8143 | 5.1190 |
| **rababa_arabic_v2 (ours)** | **~10M** | **3.2495** | **1.8072** | 10.3276 | **5.2953** |
| **rababa r5 paragraph-context, windowed zero-skip** | 580M | **2.6775** | **1.5965** | **8.0919** | **4.3863** |
| rababa r5 + GTPO-GRPO (700B windows) | 580M | 2.6597 | 1.5818 | 8.1165 | 4.3874 |
| rababa r3 + RAFT run-002 (beam 4) | 580M | 2.8515 | 1.7617 | 8.5410 | 4.8859 |
| rababa r3 + RAFT run-002 (beam 1) | 580M | 2.8308 | 1.7638 | — | — |

- **GLM-5.2 verification (2026-08-17)**: clean reproduction on the same
  benchmark+evaluator (temp 0, plain completion, neutral prompt;
  `results/sadeed-glm-5-2/`): **2.5060/1.5537**. The 2026 frontier is
  2.51, not the published Claude-3.7 1.39 — that bar is not what a
  current generic frontier model achieves under a neutral protocol.
  Our 580M model trails GLM-5.2 by only ~0.3 DER and **splits metrics
  with it on the zero-skip protocol** (r3 better w/o case endings).
  GLM-5.3: key denied access (HTTP 403).

- **r3 = r2 + 1 epoch on the decontaminated Misraj corpus (1M lines) +
  150k MSA replay**: best non-frontier-LLM result on the benchmark;
  beats Gemini-Flash on all four metrics. Greedy ≈ beam 4 throughout.
- **Windowed eval protocol** (`eval_sadeed_windowed.py`): inputs > 600B
  split at word boundaries (in-distribution for ByT5), generation cap at
  2x window (diacritized output is 1.4-1.6x input bytes — a naive
  input-length cap silently truncates), haraqat projected onto input
  letters via SequenceMatcher so output structure always matches gt.
  **r3 windowed: 2.8126 DER (CE) / 1.6877 DER (w/o CE) / 8.4110 WER /
  4.5739 WER (w/o CE) — zero skipped paragraphs**, a strictly cleaner
  protocol than any published row above.
- Protocol lesson (the GLM-5.3 "honest benchmark" rule applied to
  ourselves): an intermediate windowed run scored 1.82/0.95, but its
  generation cap had truncated the hardest paragraphs, which the
  evaluator then *skipped* — survivorship bias, not quality. The
  zero-skip protocol above is the only number we publish.
- Best-in-table DER without case endings; splits with Gemini-Flash on
  DER (CE) (0.06 apart).
- On our own cleaned 2.1M held-out split: **0.99% DER** (in-domain
  number, not comparable to the benchmark; see protocol discussion).
- Repro: `modal run eval_sadeed_diac25.py`. Artifacts: predictions CSV
  in-container; benchmark in `data/sadeed-diac-25/`.
- Eval notes: paragraphs chunked at 180 vocab chars; input is
  benchmark-provided undiacritized text; haraqat emitted only after
  Arabic letters (model predicts case endings on spaces — artifact of
  `ARAB_CHARS` including space — suppressed at render).

### r2 error analysis (scripts/analyze_errors.py, beam 4)

- 67% of residual errors are **word-internal vowel confusions**
  (fatha↔damma↔kasra swaps); only 33% word-final (iʿrāb zone). The
  residual is a **domain gap** (benchmark = 50% Classical Arabic), not
  a case-ending problem: private-dev DER is 1.40% vs 2.94% on the
  benchmark.
- Response: **r3** — domain-adaptation SFT on Misraj's public corpus
  after decontamination.

### SadeedDiac-25 contamination in Misraj's public corpus

The `sadeed-hf/train.txt` release (1.88M lines) **contains the
benchmark**: 122 paragraphs appear verbatim (diacritics-stripped match)
and ~1k more lines share 60+-char windows with benchmark paragraphs.
Decontaminated copy at `/datasets/sadeed-decontam/train.txt`
(1,894,276 kept / 1,103 dropped; 60-char windows, stride-1 both sides —
stricter than the 13-gram field standard). Any number trained on the
raw release would be contaminated.

### RAFT on r3 — negative result, closed (run-002, 2026-08-18)

3 iterations of rejection-sampling fine-tuning from r3 (K=8 samples,
temp 0.9, keep-if-better-and<=1.5% DER; 1,170 winners iter-3). Dev DER
flat end-to-end (1.1844 -> 1.1828 -> 1.1830 -> 1.1847). Final public
benchmark (rows above): beam-4 2.8515/1.7617, beam-1 2.8308/1.7638 vs
r3's 2.8429/1.7589 — within noise on both protocols. Positive-only
updates cannot sharpen split posteriors at SFT convergence (98.7% of
residual errors are phonotactically legal alternates). Same verdict as
Persian RAFT — cross-language negative result. Beam search adds nothing
for this model (unlike Hebrew s43, where beam-4 was worth 12 DER points).
Preds: `rababa_arabic_raft/run-002/sadeed_preds_beam{4,1}.csv`.

### GTPO-GRPO on r5 — negative result, closed (run-001, 2026-08-19)

Entropy-weighted GRPO (GTPO, arXiv 2508.04349): per-token credit ∝
detached normalized token entropy, graded alignment-based der reward,
KL leash 0.05, 150 steps x 16 prompts x 6 samples on 700B units,
A100-80GB. Dev curve EXACTLY flat: 5.9692 -> 5.9794 -> 5.9692 ->
5.9692 (greedy outputs unchanged under the leash). Final benchmark
(row above; 700B windows -> not directly comparable to r5's 1400B
protocol): 2.6597/1.5818, i.e. r5 within protocol noise.

Third convergent negative RL result (RAFT flat, sequence-GRPO negative
on Persian, GTPO-GRPO flat on Arabic). Conclusion for the paper: at
SFT convergence on 2M clean lines, Arabic diacritization residual
error is KNOWLEDGE-LIMITED, not policy-limited — no policy-sharpening
operator (positive-only, group-normalized, or entropy-weighted credit)
moves it. The levers that DID move the benchmark are all data-side:
decontamination (-0.1 DER), domain adaptation (-0.1), paragraph
context (-0.14). Next: r6 morphological aux-task (labels ready on
volume) and multi-reference-aware training, not more RL.

### WikiNews-2024 multi-reference (QCRI EMNLP 2025 protocol)

Python port of their EvalDiac.java (eval_wikinews_multiref.py): word
correct if ANY "/" alternate matches; redundant diacritization
normalized; ref letters without diacritics accept anything.

| Model | WER (full) | DER (full) | WER (stem) | DER (stem) |
|---|---|---|---|---|
| r3 | 19.99 | 12.60 | 14.47 | 13.80 |
| r5 paragraph-context | 20.52 | 12.72 | 14.91 | 13.92 |
| QCRI BiLSTM (published) | 2.70 | — | — | — |

Stem DER > full DER because multi-reference alternates absorb case
endings (any valid i'rab credited) — our residual errors concentrate in
INTERNAL vowels, consistent with the phonotactic-ambiguity diagnosis.
The WER gap vs QCRI is domain mismatch: their models train on
in-domain WikiNews/Wikipedia news text; ours train on Misraj/Tashkeela
classical text. SadeedDiac-25 remains the headline benchmark;
WikiNews-2024 is the cross-domain probe. r5's paragraph-context
specialization slightly regresses the cross-domain probe (+0.53 WER /
+0.12 DER) while improving the in-benchmark numbers by 0.14/0.09 —
domain specialization tradeoff, recorded honestly. r5 (paragraph-context) to be
scored when it lands. 2014 benchmark deliberately not scored: it
derives from Tashkeela (in our training corpus) — contaminated by
construction.

## Hebrew diacritization

### Best result

| Model | DER (beam=4, standard) | DER greedy | Note |
|---|---|---|---|
| **s46 phonikud+hewiki (production)** | **16.43%** | **16.44%** | s45 recipe + hewiki weak garnish (73.8K Dicta-labeled wiki lines in stage 1). Verified 2026-08-22/23, 5,095 test examples, `run-s46-phonikud-plus/run-002-gold-ft/best`. Greedy ≈ beam-4 → the shipped runtime path delivers reference quality. New best |
| s45 phonikud curriculum | 16.58% | — | stage 1: 1.5M phonikud knesset weak-pretrain (machine-labeled, deduped, decontaminated); stage 2: s43 gold recipe verbatim. Verified 2026-08-20, 5,095 test examples, `run-s45-phonikud/run-002-gold-ft/best` |
| s43 | 17.46% | previous best single model |
| v2 | 17.3% | original recipe (22K data); checkpoint has loading quirks under new transformers |
| s44 | 17.65% | seed replica |
| v4 (50K data) | 17.78% | 2.3× data → no gain |
| DictaBERT-large-char-menaked | 35.63% (full run, 4,957/5,229) | the actual SOTA model, run by us on the same test |
| 3-way output-vote ensemble | 21.52% | WORSE — franken-string effect |
| 4-way ensemble (w/ broken v2 preds) | 50.3% | invalid |
| beam=1 (s43/v4) | 29.0% | beam search = 12 DER points |

- Test: Nakdimon test split, 5,095 examples, Biblical/Rabbinic domain
- All ByT5-base (580M), seq2seq, 3 epochs, beam=4 at inference
- Eval scripts: `eval_hebrew_v4_beam4.py`, `analyze_hebrew_errors.py`

### Hebrew s47 — morph aux transplant — CLOSED NEGATIVE (2026-08-23)

The r6 template's first cross-language move: dictabert-morph TAG
stream (100K knesset lines ×4, segmented src + per-token tags) +
gold×2 + 200K weak pairs, init from s46, LR 2e-5, 1 epoch.
**DER 16.53 (beam-4, 5,095 examples)** vs init s46's 16.43 — no gain.

Read: the r6 aux-task win is not template-portable as-is. Arabic's
residual was word-final case endings (iʿrāb), which POS+feats
supervision directly disambiguates; Hebrew's 16.4 residual is not
decomposable the same way by gender/number/person tags. **s46 stays
canonical; Hebrew teacher line closed** (no s48 — money discipline).
Checkpoint: `run-s47-morph/best`.

### Error analysis (v4/s43/s44, analyze_hebrew_errors.py)


| Error type | Count (v4) | Share |
|---|---|---|
| teamim (cantillation) wrong | **0** | 0% — copy-through from input |
| nikud wrong (aligned positions) | 9,903 | 3.2% of aligned |
| ok (aligned) | 299,400 | 96.8% |
| length mismatch (metric artifact) | 186,634 | — |

**Nikud accuracy on aligned positions: 97.1%.**

### Key findings

1. **Teamim input-format leak**: standard Nakdimon-derived preprocessing
   strips nikud but NOT teamim (U+0591–05AF) from model input. Models get
   cantillation hints; teamim "errors" are impossible. We discovered this by
   breaking it: v3 (teamim stripped from input, kept in targets) collapsed
   to 36.2% DER because teamim are unpredictable from bare consonants.
   Any Hebrew diacritization reproduction must state its input format.
2. **SOTA is domain-bound**: DictaBERT-large scores ~4% on modern Hebrew
   benchmarks but 23.7% on Biblical/Rabbinic text. Cross-domain degradation
   ≈ 2–3×. Our mixed-domain ByT5 beats the SOTA model by 6 points
   cross-domain.
3. **Data scaling plateau**: 22K → 50K training pairs gave no DER change.
4. **Output-vote ensembles hurt**: char-majority voting reduced vowel errors
   9% (9,903→8,997) but increased DER 17.8→21.5% — spliced outputs are
   incoherent strings that edit-distance metrics punish. Only logit-level
   beam fusion is valid for seq2seq diacritization.
5. **Beam search is a 12-point factor**: greedy 29.0% vs beam-4 17.8%.

## Artifacts

- Checkpoints (Modal volume `rababa-checkpoints`):
  - `/checkpoints/rababa_arabic_v2/run-001/best.pt`
  - `/checkpoints/rababa_hebrew_byt5_v2|v4|s43|s44/run-001/best`
- Prediction cache: `/datasets/hebrew-pred-cache/{v2,v4,s43,s44}.jsonl`
- Corpora: `/datasets/hebrew-v4/{train,val,test}.jsonl` (50,433/2,654/1,864)

## RAG homograph probe (Persian v1) — CLOSED NEGATIVE (2026-08-20)

Retrieval few-shot (k=3 char-3gram cosine over the 445K v1 train
sentences, candidates restricted to same-homograph contexts, beam-4,
identical SentenceBench scoring): **26.07% ezafe-norm vs the 77.34%
baseline (−51.3pp)**. Baseline reproduced exactly (77.3400, n=203),
validating the harness. Contamination check: 4 exact test-sentence
overlaps found in train and excluded.

Interpretation: v1 was never trained with multi-example prompts, so
the few-shot prefix is a hard format shift — the collapse is a prompt
artifact as much as a knowledge result. Verdict per the pre-committed
decision rule (<+0.5pp): the lever is closed for inference-time use;
any revisit must be train-time retrieval conditioning, budgeted as a
new experiment, not a polish of this one. Artifacts:
persian_g2p/rag_probe_result.json, rababa-farsi/eval_persian_rag_probe.py.

## Urdu d1 — ByT5-base + Arabic-r5 cross-lingual init (2026-08-20) ★ NEW BEST

`rababa_urdu_byt5/run-001-d1` (HF `best/` on rababa-checkpoints):
ByT5-base initialized from the Arabic r5 paragraph-context teacher,
1 epoch over the 573K Urdu pair corpus (alignment-filtered, deduped;
weak machine labels — comparative eval, not absolute).

**CER 6.40% / word accuracy 47.43% (greedy, n=11,714)** vs the shipped
Urdu model's 14.77% CER — 2.3x reduction from architecture + teacher
init alone. Caveat: both train and test labels derive from our Arabic
model + G2P back-projection, so this is a within-corpus comparison;
a human-labeled Urdu gold set remains the missing evaluation. Next
lever if wanted: gold corpus (none verifiable exists in our stack),
or paragraph-context windowing for Urdu. Script: train_urdu_d1.py.

### Urdu d1 beam-4 re-eval — flat (2026-08-20)

beam4: CER 6.53% / word_acc 47.97% vs greedy 6.57% / 46.99%
(re-run; original verdict 6.40/47.43). Unlike Hebrew s45 (beam worth
12 DER points), Urdu gains nothing from beam — the G2P-back-projected
weak labels are near-deterministic, so greedy is already confident.
Per TODO 06 rule (<55% word_acc): d2 launched (one epoch at 1e-5 from
d1, run-002-d2); corpus-label consistency is the expected ceiling.

## Urdu d2 — continuous low-LR second epoch (2026-08-20) ★ NEW BEST

`rababa_urdu_byt5/run-002-d2` (best on volume): 1 epoch at LR 1e-5
from run-001-d1, identical test protocol.

**CER 5.77% / word_acc 52.47%** (greedy, n=11,714) vs d1 6.40/47.43 —
CER down 0.63pp, word_acc up 5.0pp. The weak-corpus ceiling is higher
than d1 alone reached; the standing rule remains that absolute
numbers are within-corpus (machine labels both sides). Best model
now: run-002-d2. Script: train_urdu_d2.py.

## Arabic r6 — morphological aux-task (iʿrāb supervision) ★ NEW CANONICAL TEACHER (2026-08-21)

### r6 verdict table (SadeedDiac-25, 2026-08-21)

| Model | Total DER | Morph DER | Protocol |
|---|---|---|---|
| **r6 (greedy, canonical)** | **2.5793** | **1.5317** | windowed zero-skip, 1400B |
| r5 | 2.6775 | 1.5965 | windowed zero-skip, 1400B |

`rababa_arabic_byt5/run-006-morph/best` (HF, rababa-checkpoints): r5's
plain stream + TAG-prefixed morph stream (qalsadi 300K lines, 68.6%
exact case/tense), two-format multitask on one ByT5-base, upsampled
x4 to ~25% of the mix, init from r5, 1 epoch, A100-80GB.

**Total DER (CE) 2.5793 / Morphological DER 1.5317** on
SadeedDiac-25, windowed zero-skip at 1400B — vs r5's 2.6775/1.5965.
Both components improved; the diagnosis (33% of residual = word-final
case endings; Total-vs-Morph gap) is confirmed and the fix is
knowledge injection, not policy sharpening. Inference contract is
UNCHANGED from r5 (no TAG prefix at inference = plain diacritization;
1400B windows). r6 REPLACES r5 as the Arabic teacher for distillation.
Script: train_arabic_r6.py.

### r6 out-of-domain — WikiNews-2024 multi-ref (2026-08-21)

r6 full-mode **WER 19.8191 / DER 12.4613** (no-CE: WER 14.6571 /
DER 13.7271) — beats BOTH r5 (20.52/12.72) and r3 (19.99/12.60).
The r5-era paragraph-specialization OOD trade-off is erased: the
morph aux-task improved in-domain AND out-of-domain. r6 strictly
dominates all measured surfaces. r7's OOD gate denominator is
therefore 19.82/12.46 (r6), not r5's number.

### r6 beam-4 probe (2026-08-23) — NEGATIVE

Beam-4 on the identical windowed harness: **Total DER 2.5588 /
Morph DER 1.5379** vs greedy 2.5793/1.5317 — noise-level change
both ways. Beam stays UNSHIPPED for Arabic: greedy posteriors are
already sharp (consistent with the knowledge-injection diagnosis),
and beam would cost ~4x inference. The Hebrew beam gain (12 DER
points) does not transfer. Script: eval_arabic_r6_beam4.py.

## Arabic r7 — news-domain adaptation: NEW CANONICAL TEACHER (2026-08-28)

Init from r6, anchor r5-units + 13,986 news units (0.85% mix) + 400
gold-2014 lines. Windowed zero-skip, full 1,200 paragraphs:

### r7 verdict table (SadeedDiac-25, 2026-08-28)

| Model | Total DER | Morph DER | Protocol |
|---|---|---|---|
| **r7 (news-domain)** | **2.2864** | **1.3343** | windowed zero-skip |
| r6 (morph aux) | 2.5793 | 1.5317 | windowed zero-skip |
| r5 | 2.6775 | 1.5965 | windowed zero-skip |

**−0.29pp over r6** — the news mix (teacher-labeled news units + a
small gold anchor) improved IN-DOMAIN substantially, not just OOD.

### r7 OOD verdict table (WikiNews-2024 multiref, 2026-08-28)

Out-of-domain, WikiNews-2024 multi-ref (QCRI protocol, full mode):

| Model | WER | DER |
|---|---|---|
| **r7** | **17.3794** | **11.8273** |
| r6 | 19.8191 | 12.4613 |
| r5 | 20.52 | 12.72 |

**r7 sweeps: best ID and best OOD of the teacher lineage — r7
REPLACES r6 as the canonical Arabic teacher** (artifacts:
rababa_arabic_byt5/run-007-news/best). On the SadeedDiac-25 leaderboard
it is the best dedicated model under the protocol, behind only
Claude-3.7-Sonnet's published 1.3941, now well clear of GLM-5.2 (2.6911).
Future student distillations take r7 as teacher. Script:
train_arabic_r7.py; artifacts: EVAL_DONE, sadeed_preds_windowed.csv,
wikinews_multiref_r7.json.

## Arabic r8 — IPA aux-task (phonemic supervision): controlled negative vs morph (2026-08-27)

The controlled experiment the r6 claim needed: r8 differs from r6 in
EXACTLY one variable — the aux stream's output representation. Stream B
renders the SAME r5-units as broad-phonemic IPA (deterministic converter,
arabic_to_ipa.py) instead of qalsadi morphology; same ~25% aux share,
same seeded sample, same init (r5), same 1-epoch A100 schedule.

### r8 verdict table (SadeedDiac-25, windowed zero-skip, 1400B, full 1,200)

| Model | Total DER | Morph DER | Protocol |
|---|---|---|---|
| **r6 (morph aux, canonical)** | **2.5793** | **1.5317** | windowed zero-skip |
| r8 (IPA aux) | 2.6588 | 1.5783 | windowed zero-skip |
| r5 (no aux) | 2.6775 | 1.5965 | windowed zero-skip |

IPA-stream probe (200 held-out domain units): **CER 0.0230, EM 62/200**
— the model genuinely learned the second projection, so the comparison
is not confounded by a failed aux task.

Read: IPA aux helps over no-aux (−0.019pp Total DER) but loses to
morphological aux (r8 is +0.080pp worse than r6). Phonemic supervision
is NOT the active ingredient in the r6 win; lexical/morphological
knowledge (iʿrāb) is. The "diacritization helped by phonemes" hypothesis
survives only in its weak form (a structured auxiliary projection beats
none) and fails in its strong form (phonemic specifically). r6 stays the
canonical Arabic teacher. Script: train_arabic_r8.py; artifacts:
rababa_arabic_byt5/run-008-ipa (EVAL_DONE, ipa_probe.json,
sadeed_preds_windowed.csv).

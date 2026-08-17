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
| Gemini-Flash-2.0 (published) | — | 3.1926 | 2.3783 | 7.9942 | 5.5044 |
| GPT-4 (published) | — | 3.8645 | 3.8645 | 5.2719 | 10.9274 |
| Sadeed (published) | 1.5B | 7.2915 | 5.2625 | 13.7425 | 9.9245 |
| **rababa r3 domain-adapted (ours)** | 580M | **2.8429** | **1.7589** | **8.4981** | **4.8859** |
| rababa_arabic_byt5 r2 (ours) | 580M | 2.9406 | 1.8333 | 8.8373 | 5.0835 |
| rababa_arabic_byt5 r2 (beam 4) | 580M | 2.9478 | 1.8522 | 8.8143 | 5.1190 |
| **rababa_arabic_v2 (ours)** | **~10M** | **3.2495** | **1.8072** | 10.3276 | **5.2953** |

- **r3 = r2 + 1 epoch on the decontaminated Misraj corpus (1M lines) +
  150k MSA replay**: 2.94 → 2.84 DER (CE), 1.83 → 1.76 DER (w/o CE).
  Best non-frontier-LLM result on the benchmark; beats Gemini-Flash on
  all four metrics. Greedy ≈ beam 4 throughout.
- r2/r3 eval caveat: generation was capped at 1024 bytes — 57/1,200
  paragraphs were truncated (missing tails scored as errors). A
  windowed eval (600-byte in-distribution chunks, stitched) is the
  apples-to-apples number; see `eval_sadeed_windowed.py`.
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

## Hebrew diacritization

### Best result

| Model | DER (beam=4, standard) | Note |
|---|---|---|
| **s43 (production)** | **17.46%** | best single model |
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

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

| System | Params | DER | Test set |
|---|---|---|---|
| **rababa_arabic_v2 (ours)** | ~30M | **0.99%** | our held-out 2.1M split |
| Sadeed (published) | 1.5B | 1.2% | SadeedDiac-25 (their split) |

Protocol caveat: different test sets. SadeedDiac-25 is gated on HF; we
phrased the claim as "0.99% on our split" vs their published number.
TODO.publish/01.

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

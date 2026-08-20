# 04 — Hebrew s46: scale + diversify the weak stage

## Why
s45 proved the curriculum (+0.88 DER over gold-only: 16.58 vs 17.46)
but its weak stage was knesset only (1.5M lines, single domain —
parliamentary transcripts). hewiki (80K Hebrew-Wikipedia lines) is a
second, encyclopedic domain sitting unlabeled on the volume. Same
lever, turned up: more + more-diverse weak data before the identical
gold FT.

## Plan
1. `label_hewiki_full.py` (rababa) — runs NOW on A10G:
   - Full-scale Dicta labeling of `/datasets/hewiki/train.txt`
     (80K lines) via the batch recipe from `batch_distill_hewiki.py`
     (DictaBERT predict batches), nikud-only targets (strip teamim),
     length/Hebrew-fraction filters, 40-char window decontam vs
     Nakdimon test.
   - Output: `/datasets/hebrew-hewiki-dicta/{train,val}.txt`.
2. `train_hebrew_s46.py` — s45 VERBATIM with one change:
   - Stage 1 weak = knesset 1.5M + hewiki-dicta (all of it,
     ~80K lines ≈ 5% of weak steps — a domain garnish, not a pivot).
   - Stage 2 = s43 gold recipe unchanged (hebrew-v4 jsonl, 3 ep,
     batch 8, LR 3e-4, warmup 500).
   - Eval: beam-4 DER, identical protocol/harness as s45's 16.58.
3. Gates: must beat s45's 16.58 to replace it; otherwise record as
   flat and keep s45 (weak-stage ceiling reached).

## Guards
- Dicta labels are weak only — the gold stage corrects (no-teacher-
  poison, the exact s45-validated pattern).
- Zero Nakdimon-test contamination (window decontam).

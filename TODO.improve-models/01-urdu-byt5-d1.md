# 01 — Urdu diacritization d1: ByT5-base + cross-lingual init

## Why
Urdu is our weakest shipped model (14.77% CER, urdu_diacrit/run-001,
custom char encoder trained on 635K cross-lingually machine-labeled
lines). Arabic — same script family, same task shape — sits at 2.68 DER
on ByT5-base with paragraph context. The gap is architecture + teacher
vintage, not task difficulty.

## Plan
1. Data: existing corpus on volume `urdu-diacrit-datasets`:
   - `urdu-diacritized/{train,val,test}.txt` (635K machine-labeled —
     WEAK labels, teacher-poison rule applies: treat as stage-1 only)
   - `urdu-diacrit/*.jsonl` (HF G2P-derived pairs from WO #306)
2. `train_urdu_d1.py` (rababa):
   - Init: Arabic r5 teacher `/checkpoints/rababa_arabic_byt5/
     run-005-context/best` (ByT5-base) — cross-lingual init gives the
     shared-abjad prior instead of starting cold.
   - Stage 1: 1 epoch over the weak 635K (line units, byte tokenizer).
   - Stage 2: none yet (no gold corpus found — UDD has no verifiable
     public URL; do NOT fabricate one). The run's own test split is
     machine-labeled too, so the eval is comparative, not absolute.
3. Eval: greedy CER via editdistance on `urdu-diacritized/test.txt`,
   identical protocol to the 14.77 number + word-level accuracy.
   Target: clearly under 14.77 CER (architecture+init upgrade).
4. Launch detached under supervisor app `rababa-urdu-d1`
   (EVAL_DONE guard, checkpoint-resume, volume commits).

## Guards
- No LLM teacher. No RL. Weak corpus is from our Arabic model only.
- Parallel to r6 (separate Modal app + GPU).

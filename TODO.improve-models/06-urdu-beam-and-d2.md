# 06 — Urdu d1 follow-up: beam-4 eval, then d2 decision

## Why
d1 (CER 6.40%, word_acc 47.43%) was evaluated GREEDY only. The Hebrew
s45 experience: beam-4 at inference is worth double-digit DER points
on ByT5 diacritizers. Before spending a d2 training run, collect the
free win and re-read the saturation point.

## Plan
1. `eval_urdu_d1_beam.py` (rababa): beam-4 vs greedy on the identical
   test protocol (urdu-diacrit/test.jsonl, CER + word_acc, n=11,714).
2. Decision after the number:
   - If beam-4 word_acc still < ~55%: d2 with a second epoch at lower
     LR (1e-5) from run-001-d1 — cheap, single A100 pass.
   - If beam-4 lifts word_acc >= ~55%: declare d1 final for the weak
     corpus; the next real gain requires gold Urdu data (none
     verifiable — do not chase it).

## Guards
- Same test set, same alignment protocol as the d1 verdict so the
  numbers are comparable line-for-line.

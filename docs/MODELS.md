# Model manifest — canonical checkpoints for downstream distillation

For the agent distilling our models into mini-models: this file is the
source of truth for which checkpoint to use as teacher. Updated as
verdicts land; check `git log` on this file for recency.

## Arabic diacritization (SadeedDiac-25, Misraj evaluator)

| Teacher | Volume path | DER (CE) / (w/o CE) | Protocol | Status |
|---|---|---|---|---|
| **r5 paragraph-context (current best)** | `/checkpoints/rababa_arabic_byt5/run-005-context/best` | **2.6775 / 1.5965** | windowed zero-skip, 1400B | ✅ final |
| r3 | `/checkpoints/rababa_arabic_byt5/run-003-domain/best` | 2.8126 / 1.6877 | windowed zero-skip, 600B | final |
| r3 raw | same | 2.8429 / 1.7723 | single-shot 1024 | final |
| r2 | `/checkpoints/rababa_arabic_byt5/run-002-full-2ep/best` | 2.9406 / 1.8333 | single-shot | final |
| char-encoder (10M) | `/checkpoints/rababa_arabic_v2/run-001/best.pt` | 3.2495 / 1.8072 | benchmark protocol | final |

| GTPO-GRPO on r5 | `/checkpoints/rababa_arabic_grpo/run-001/best` | 2.6597 / 1.5818 | 700B windows (≠ r5's 1400B) | final — flat vs r5, kept for reference only |
| RAFT-002 on r3 | `/checkpoints/rababa_arabic_raft/run-002/best` | 2.8515 / 1.7617 | benchmark, beam 4 | final — flat vs r3 |

Teacher stays **r5**: both RL variants were flat/negative, so r5's best
is the distilled-knowledge ceiling. (GLM-5.2 is NOT a teacher: standing
rule — no LLM teachers for diacritization; see
`results/sadeed-glm-5-2/README.md` for why the distillation idea was
rejected on principle.)

## Evaluation protocol for parity gates

- Benchmark: `data/sadeed-diac-25/train.parquet`, evaluator:
  `sadeed_evaluator.py` (Misraj, default, gt_missing_diacritic_is_error=False).
- Zero-skip windowed protocol (use for parity vs teacher):
  `eval_sadeed_windowed.py` — split >600B inputs at word boundaries,
  generation cap = 2x window (diacritized output is 1.4–1.6x input
  bytes; input-length caps silently truncate), haraqat projected onto
  input letters. A distilled model's parity target is the teacher's
  zero-skip number within tolerance.

## Other languages (best checkpoints)

- Hebrew diacritization: ByT5-base **s45 phonikud curriculum, 16.58% DER**
  (prior s43: 17.46; Nakdimon Biblical
  test) — see `docs/RESULTS.md`.
- Persian G2P: `persian_g2p/run-001/best` (v1, 77.34% SB HA ezafe-norm;
  RAFT run tied — v1 remains canonical).
- Thai G2P: umt5 continued-fine-tune + epitran augmentation at scale
  (367K lines, full Wikipedia), **1.7260% PER** (prior 2.32%; public
  baseline 6.37%) — `/ckpts/secryst_thai_ipa_scaleup600k/run-001/best`
  on `secryst-checkpoints`; secryst `docs/paper-thai`.
- Khmer: secryst forward model (59.66% EM) + backward (50.3% EM).

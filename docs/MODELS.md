# Model manifest — canonical checkpoints for downstream distillation

For the agent distilling our models into mini-models: this file is the
source of truth for which checkpoint to use as teacher. Updated as
verdicts land; check `git log` on this file for recency.

## Arabic diacritization (SadeedDiac-25, Misraj evaluator)

| Teacher | Volume path | DER (CE) / (w/o CE) | Protocol | Status |
|---|---|---|---|---|
| **r6 morph aux (current best, CANONICAL)** | `/checkpoints/rababa_arabic_byt5/run-006-morph/best` | **2.5793 / 1.5317** | windowed zero-skip, 1400B, greedy | ✅ final |
| r5 paragraph-context | `/checkpoints/rababa_arabic_byt5/run-005-context/best` | 2.6775 / 1.5965 | windowed zero-skip, 1400B | superseded by r6 |
| r3 | `/checkpoints/rababa_arabic_byt5/run-003-domain/best` | 2.8126 / 1.6877 | windowed zero-skip, 600B | final |
| r2 | `/checkpoints/rababa_arabic_byt5/run-002-full-2ep/best` | 2.9406 / 1.8333 | single-shot | final |
| char-encoder (10M) | `/checkpoints/rababa_arabic_v2/run-001/best.pt` | 3.2495 / 1.8072 | benchmark protocol | final |

r6 = plain + "TAG: "-prefixed morph aux streams (qalsadi 300K lines,
iʿrāb supervision), init from r5. It also wins OOD (WikiNews-2024
multi-ref: WER 19.82 / DER 12.46 vs r5 20.52/12.72), so it strictly
dominates every measured surface. Beam-4 probed 2026-08-23: flat —
greedy IS the contract (4x cheaper, same quality). Distill from r6;
the morph stream never appears at inference (no TAG prefix = plain
diacritization). (Standing rule unchanged: no LLM teachers for
diacritization — see `results/sadeed-glm-5-2/README.md`.)

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

- Hebrew diacritization: ByT5-base **s46 phonikud+hewiki, 16.43% DER**
  (s45: 16.58; s43: 17.46; Nakdimon test, beam-4; greedy 16.44 ≈
  beam) —
  `/checkpoints/rababa_hebrew/run-s46-phonikud-plus/run-002-gold-ft/best`.
  s47 (morph aux transplant) CLOSED NEGATIVE 2026-08-23: 16.53 vs
  16.43 — the r6 template is not portable as-is; teacher line closed.
- Urdu diacritization: **urd-diac-1.0 (shipped, ByT5-small) is the
  champion — CER 3.74 / word_acc 67.51** on urdu-diacrit/test.jsonl
  (comparable eval 2026-08-23, both models under the 5.14.1 export
  stack). d2 (`rababa_urdu_byt5/run-002-d2`, ByT5-base) measured
  5.94/51.95 on the SAME harness — it is the rababa-lineage best
  only, not the cross-lineage best; do not ship it over 1.0.
  Version trap: the urd-diac-1.0 checkpoint generates EMPTY strings
  under transformers 4.46.3 — evaluate legacy checkpoints under their
  export stack (torch 2.12.1 / transformers 5.14.1). No public gold;
  CLE Pakistan inquiry pending.
- Persian G2P: `persian_g2p/run-001/best` (v1, 77.34% SB HA ezafe-norm;
  RAFT run tied — v1 remains canonical). FROZEN — no more training.
- Thai G2P: umt5 continued-fine-tune, **1.7260% PER** (public baseline
  6.37%) — `/ckpts/secryst_thai_ipa_scaleup600k/run-001/best` on
  `secryst-checkpoints`. FROZEN.
- Khmer G2P: v1 full-orthography 58.99% word_acc / 26.90 CER
  (`khmer_g2p_byt5/run-001/best`, secryst-checkpoints; shipped as
  khm-latn-1.0) + v2 vowel-restoration run-002 (stripped-input
  19.22% word_acc where rules score 0.0%; full-surface 53.18%).
  Ship rule: rules/v1 for full orthography, v2 for reduced input.

# Source models for distillation — usage prompt (2026-08-19 refresh)

You are distilling the Interscript ML source models into client-tier
mini-models. The teacher manifest is `interscript/rababa/docs/MODELS.md`
(source of truth; verify recency via `git log`). This file is the
usage contract per model.

General rules (unchanged):
- Teachers are HF-format `best/` dirs on Modal volume `rababa-checkpoints`
  (except where noted). Copy with `modal volume get`, never re-export
  from training containers.
- Parity gate per model: student must land within +5pp of the teacher on
  the same harness, or be rejected. Never ship a student you didn't measure.
- No LLM is ever a teacher (standing project rule).
- NEW standing verdict: RL variants (RAFT/GRPO/GTPO) are all flat or
  negative on these teachers — do not retrain or RL-polish a teacher,
  and treat teacher outputs (not any RL variant) as the label source.
  Data-side knowledge injection WORKS: r6's morph aux-task is the
  template (aux supervision >> policy sharpening).
- r6 VERDICT LANDED (2026-08-21): 2.5793/1.5317, replaces r5. The
  "in flight" caveat is void.
- SUCCESSORS IN FLIGHT (2026-08-20): r6 Arabic morph aux-task
  (run-006-morph, resumed from checkpoint-6000 after a preemption;
  ~15h remaining). Standing rule: it does not replace r5 until a
  verified eval lands in docs/RESULTS.md AND this file is updated.
  Do not poll, do not use intermediate checkpoints. (VERIFIED:
  Thai scaleup600k 1.7260% PER — section 4; Hebrew s45 16.58% DER —
  section 2.)

## 1. Arabic diacritizer — r6 morph aux-task (580M ByT5-base) ★ CANONICAL (2026-08-21)
- Path: `/checkpoints/rababa_arabic_byt5/run-006-morph/best` (HF)
- Quality: **2.5793 DER (CE) / 1.5317 (w/o CE)** on SadeedDiac-25,
  windowed zero-skip at 1400B — beats r5 (2.6775/1.5965). VERIFIED
  2026-08-21; r6 replaces r5 per this update.
- Contract: input undiacritized Arabic text → output same text with
  haraqat. Context matters: feed up to 1400 bytes per call and window
  longer documents at word boundaries (see `eval_sadeed_windowed.py`).
  Generation cap = 2x window bytes (diacritized output is 1.4–1.6x
  input). For short inputs (≤600B) behavior matches r3.
- Parity harness: Misraj evaluator (`sadeed_evaluator.py`) on the full
  1,200-paragraph benchmark, zero skips. Student target: ≤ 3.07 DER
  (CE) windowed (= teacher +5pp-equivalent; scale from 2.58).
- `run-005-context/best` remains available as the fallback teacher;
  r6's inference contract is identical (no TAG prefix at inference). GTPO-GRPO run-001 was flat
  vs r5 — do NOT use it as teacher. The 10M char-encoder
  (`rababa_arabic_v2/run-001/best.pt`, 3.2495/1.8072) remains the
  embedded-tier teacher if the student must be tiny.
- Out-of-domain probe (WikiNews-2024 multi-reference, QCRI protocol):
  r3 19.99/12.60, r5 20.52/12.72 — paragraph specialization trades
  ~0.5 DER there. If the client tier is news-heavy, flag it; do not
  silently evaluate news text on the SadeedDiac-25 harness.

## 2. Hebrew diacritizer — s45 phonikud curriculum (ByT5-base) ★ VERIFIED 2026-08-20
- Path: `/checkpoints/rababa_hebrew/run-s45-phonikud/run-002-gold-ft/best` (HF)
- Contract: input Hebrew consonants (+teamim preserved as-is) → same
  text + nikud. **Use beam=4 at inference**; greedy costs 12 DER points.
- Quality: **16.58% DER** on Nakdimon Biblical test (5,095 examples,
  identical protocol/harness as s43's 17.46 and DictaBERT's 35.6).
  Recipe: 1.5M-line phonikud knesset weak-pretrain (Dicta
  machine-labels, deduped 2.48M boilerplate dups, zero test
  contamination) THEN the s43 gold recipe verbatim — the gold stage
  corrects the teacher, per the no-teacher-poison rule.
- Distill notes: teacher labels MUST be beam-4 outputs; ~5.6pp shrink
  cost observed previously is acceptable. s43 remains available at
  `/checkpoints/rababa_hebrew/run-s43/best` if a comparison student
  is wanted.

## 3. Persian G2P — v1 (ByT5-small, RELEASE-FROZEN)
- Path on volume `persian-g2p-checkpoints`: `/checkpoints/persian_g2p/run-001/best`
- Contract: raw Persian sentence (NO prefix) → space-separated Latin
  phonemes.
- Quality: test CER ≈1.6%; SentenceBench homograph 77.34% ezafe-norm
  (published Homo-GE2PE SOTA: 76.89%).
- Two-metric gate: CER via editdistance on v1 test split AND homograph
  via `eval_sentencebench.py`. GRPO/RAFT variants scored WORSE — v1 is
  final; no further teacher polish is coming.
- v5 re-score verdict (2026-08-19): its raw 28.22% SB was a
  representation artifact; decoded with the Mapped→Plain table
  (`eval_persian_v5_rescore.py`, 114,869 entries) it reaches 53.47%
  exact / 71.29% ezafe-norm — still 6 points under v1. v5/mapped repr
  is CLOSED negative; v1 remains the only Persian teacher.

## 4. Thai G2P (umt5 continued-FT) — scaleup600k IS the teacher ★ VERIFIED 2026-08-19
- Path: `/ckpts/secryst_thai_ipa_scaleup600k/run-001/best` on volume
  `secryst-checkpoints` (modal volume get).
- Contract: Thai graphemes → IPA phonemes; beam 4.
- Quality: **1.7260% PER** on the fixed 1,219-sentence Kaikki test
  (same harness as all previous rows; tha scaleup before it: 2.32%;
  public baseline 6.37%). Verified by
  `secryst/train_thai_scaleup.py::evaluate` (metrics.json on the
  volume). Known artifact: the sentence-level WER printed as 1.0 —
  treat PER as the parity metric, investigate WER only if your student
  harness shows nonzero exact matches where this shows zero.
- Parity gate: student within +5pp-equivalent of teacher PER (scale
  from 1.73 → student must land ≤ ~2.6% PER on the same test set).

## Suggested order (2026-08-19)
Start NOW, in parallel where possible:
1. Arabic r5 (canonical teacher; biggest quality story — a mini model
   at ≤3.2 DER windowed still beats Gemini-Flash published). r6 may
   land ~20h later; if it verifies better you re-distill once — cheap
   next to idling.
2. Persian (two-metric gate; v1 is final, nothing in flight can
   replace it — the repr line of work is closed).
3. Hebrew s45 (beam-4 teacher labels required) — GO, verified 16.58.
4. Thai — GO. scaleup600k verified 1.7260% PER (section 4); smallest,
   cleanest contract, first recommended target.

## What we will NOT ship
- Any LLM-labeled variant (hallucinated haraqat poison chains).
- Any RL-polished teacher (three controlled runs, all flat/negative;
   see `rababa/docs/RESULTS.md` for the tables).

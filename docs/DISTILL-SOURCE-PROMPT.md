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
- SUCCESSORS IN FLIGHT (2026-08-19): r6 Arabic morph aux-task
  (run-006-morph, ~20h), Hebrew s45 phonikud curriculum
  (run-s45-phonikud, ~11h), Thai scaleup600k (~12h). Standing rule:
  none replaces its teacher below until a verified eval lands in
  docs/RESULTS.md AND this file is updated. Do not poll for them, do
  not use intermediate checkpoints.

## 1. Arabic diacritizer — r5 paragraph-context (580M ByT5-base) ★ NEW CANONICAL
- Path: `/checkpoints/rababa_arabic_byt5/run-005-context/best` (HF)
- Quality: **2.6775 DER (CE) / 1.5965 (w/o CE)** on SadeedDiac-25,
  windowed zero-skip at 1400B — beats our verified GLM-5.2 reproduction
  (2.6911/1.7179). This is the strongest Arabic teacher we will ship.
- Contract: input undiacritized Arabic text → output same text with
  haraqat. Context matters: feed up to 1400 bytes per call and window
  longer documents at word boundaries (see `eval_sadeed_windowed.py`).
  Generation cap = 2x window bytes (diacritized output is 1.4–1.6x
  input). For short inputs (≤600B) behavior matches r3.
- Parity harness: Misraj evaluator (`sadeed_evaluator.py`) on the full
  1,200-paragraph benchmark, zero skips. Student target: ≤ 3.18 DER
  (CE) windowed (= teacher +5pp-equivalent; scale from 2.68).
- `run-005-context/best` is RELEASE-FROZEN. GTPO-GRPO run-001 was flat
  vs r5 — do NOT use it as teacher. The 10M char-encoder
  (`rababa_arabic_v2/run-001/best.pt`, 3.2495/1.8072) remains the
  embedded-tier teacher if the student must be tiny.
- Out-of-domain probe (WikiNews-2024 multi-reference, QCRI protocol):
  r3 19.99/12.60, r5 20.52/12.72 — paragraph specialization trades
  ~0.5 DER there. If the client tier is news-heavy, flag it; do not
  silently evaluate news text on the SadeedDiac-25 harness.

## 2. Hebrew diacritizer — s43 (ByT5-base, RELEASE-FROZEN)
- Path: `/checkpoints/rababa_hebrew/run-s43/best` (HF)
- Contract: input Hebrew consonants (+teamim preserved as-is) → same
  text + nikud. **Use beam=4 at inference**; greedy costs 12 DER points.
- Quality: 17.46% DER on Nakdimon Biblical test (DictaBERT: 35.6% same
  protocol). Positioning: Biblical/rabbinic vocalization for Torah
  study, digital humanities, bibliographic transliteration.
- Distill notes: teacher labels MUST be beam-4 outputs; ~5.6pp shrink
  cost observed previously is acceptable.
- s45 phonikud curriculum is training (may beat 17.46); until its
  verified eval lands here, s43 is the teacher. Do not distill from
  phonikud/Dicta machine labels directly (teacher-poison rule).

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

## 4. Thai G2P (ByT5-small / umt5 continued-FT) — WAIT for scaleup verdict
- Path: secryst Thai stack (`secryst/docs/paper-thai` + MODELS.md).
- Contract: Thai graphemes → IPA phonemes.
- Quality: **2.32% PER** (public baseline 6.37%).
- **INSTRUCTION: do NOT start the Thai distillation yet.** The
  scaleup600k successor (367K epitran lines, 7.3x corpus) finishes
  ~12h after 2026-08-19 launch and is the ONLY in-flight run that
  should gate your start — it finishes soonest and Thai is the
  smallest job, so waiting costs half a day and may save a full
  re-distill. Check this file first: if section 4 lists a scaleup600k
  teacher with a verified PER, distill THAT; if it still names
  thai_combined_mixed, scaleup failed or was flat — distill
  thai_combined_mixed. Until then, spend the wait building the Thai
  harness (test.jsonl, beam-4 PER scorer from
  `secryst/train_thai_scaleup.py::evaluate`).

## Suggested order (2026-08-19)
Start NOW, in parallel where possible:
1. Arabic r5 (canonical teacher; biggest quality story — a mini model
   at ≤3.2 DER windowed still beats Gemini-Flash published). r6 may
   land ~20h later; if it verifies better you re-distill once — cheap
   next to idling.
2. Persian (two-metric gate; v1 is final, nothing in flight can
   replace it — the repr line of work is closed).
3. Hebrew s43 (beam-4 teacher labels required). s45 may land ~11h
   later; same re-distill-once logic as Arabic.
4. Thai — WAIT per section 4 (gate on scaleup600k verdict, then it
   becomes the first actual distillation target: smallest, cleanest
   contract).

## What we will NOT ship
- Any LLM-labeled variant (hallucinated haraqat poison chains).
- Any RL-polished teacher (three controlled runs, all flat/negative;
   see `rababa/docs/RESULTS.md` for the tables).

# Source models for distillation — usage prompt

You are distilling the Interscript ML source models into client-tier
mini-models. The teacher manifest lives at
`interscript/rababa/docs/MODELS.md` (source of truth; verify recency via
git log). This file is the usage contract per model.

General rules:
- Teachers are HF-format `best/` dirs on Modal volume `rababa-checkpoints`
  (except where noted). Copy with `modal volume get`, never re-export from
  training containers.
- All four sources below are RELEASE-FROZEN: their numbers will not
  change. Do not wait for better checkpoints.
- Parity gate per model: student must land within +5pp of the teacher on
  the same harness, or be rejected. Never ship a student you didn't
  measure.
- No LLM is ever a teacher (standing project rule).

## 1. Arabic diacritizer — char-encoder (10M, RELEASE-FROZEN)
- Path: `rababa_arabic_v2/run-001/best.pt` (raw torch state, NOT HF —
  load via `interscript/rababa/src/rababa`: `build_model(cfg)` + ArabicEncoder)
- Contract: input undiacritized Arabic string → output same string with
  haraqat after each letter. Chunk at 180 vocab chars.
- Quality: 3.2495 DER (CE) / 1.8072 (w/o CE) on SadeedDiac-25, zero skips.
- Positioning: edge/embedded tier. The ByT5 teachers (r3/r5/GRPO) are
  still moving — do NOT distill those yet; a "ara-diac-base" teacher
  freeze will be announced in MODELS.md.
- Distill target: same I/O contract, any student init; verify with
  Misraj evaluator (`sadeed_evaluator.py`), windowed zero-skip protocol.

## 2. Hebrew diacritizer — s43 (ByT5-base, RELEASE-FROZEN)
- Path: `/checkpoints/rababa_hebrew/run-s43/best` (HF format)
- Contract: input Hebrew with consonants (+teamim preserved as-is) →
  same text + nikud. Use beam=4 at inference; greedy costs 12 DER points.
- Quality: 17.46% DER on Nakdimon Biblical test (DictaBERT: 35.6%).
- Positioning: best-in-world on Biblical/rabbinic vocalization; NOT a
  modern-Hebrew full-nikud product (nobody needs that). Targets: Torah
  study tools, digital humanities, bibliographic transliteration.
- Distill notes: teacher labels for distillation MUST be beam-4 outputs;
  a prior heb-diac-small run showed ~5.6pp shrink cost — acceptable.

## 3. Persian G2P — v1 (ByT5-small, RELEASE-FROZEN)
- Path on volume `persian-g2p-checkpoints`: `/checkpoints/persian_g2p/run-001/best`
- Contract: input raw Persian sentence (NO prefix — v1 format) → output
  space-separated Latin phonemes (e.g. "besAdegi qAbele ...").
- Quality: test CER ≈1.6%; SentenceBench homograph 77.34% ezafe-norm
  (hard-slice metric; Homo-GE2PE published: 76.89%).
- Positioning: TTS frontend / transliteration source; the homograph
  number is the stress test, the CER is the product number.
- Distill: measure student on BOTH (CER via editdistance on the v1 test
  split; homograph via eval_sentencebench.py protocol).

## 4. Thai G2P (ByT5-small, RELEASE-FROZEN)
- Path: secryst Thai stack (see `secryst/docs/paper-thai` + MODELS.md).
- Contract: Thai graphemes → IPA phonemes.
- Quality: 2.32% PER (public baseline 6.37%).
- Positioning: Thai TTS frontend + transliteration; smallest and easiest
  win — recommended FIRST distillation target.

## Suggested order
1. Thai (smallest, cleanest contract, biggest headroom over baseline)
2. Persian (clear product story, two-metric gate)
3. Hebrew (beam-4 teacher labels required)
4. Arabic char-encoder (needs the rababa loader, not plain HF)

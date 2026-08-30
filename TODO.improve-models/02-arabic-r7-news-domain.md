# 02 — Arabic r7: news-domain adaptation (OOD repair)

## Why
r5 paragraph-context specialized to the SadeedDiac domain and trades
~0.5 DER out-of-domain (WikiNews-2024 multi-ref: r5 20.52/12.72 vs
r3 19.99/12.60). If r6 verifies, case endings improve too — but the
OOD gap needs domain data, not morphology.

## Plan
1. `label_arabic_news.py` (rababa) — runs NOW, parallel to r6:
   - Fetch unlabeled Arabic news from HF
     (`khalidalt/ultimate_arabic_news`, fallback
     `Abdelkareem/arabic-bbc-news`), clean (Arabic-letter fraction,
     length, dedupe).
   - Pseudo-label with r5 (windowed 1400B zero-skip, greedy, same
     harness as eval) → volume `/datasets/arabic-news-r5/`.
   - Add GOLD news: `/datasets/wikinews/WikiNews_2014.txt.diac`
     (gold-diacritized, different year/documents than the 2024 probe).
   - NEVER touch `WikiNews_2024*` — that is the OOD probe.
2. `train_arabic_r7.py` — launch AFTER r6 verdict + labels exist:
   - INIT = r6 best if r6 verifies better on SadeedDiac, else r5.
   - Mix: cached r5-units (replay, protects ID) + news units
     (2014 gold upweighted + r5-pseudo modern news), news ≈ 15-20%
     of steps. r5-proven batch/accum, A100-80GB.
   - Gates: SadeedDiac-25 windowed zero-skip must not regress beyond
     +0.1 DER of the init model; WikiNews-2024 multi-ref must improve.
3. Fold verdict into docs/RESULTS.md + DISTILL-SOURCE-PROMPT.md only
   when both gates pass.

## Guards
- Domain adaptation via pseudo-labels is self-training — the replay
  majority + gold 2014 news keeps ID anchored; the ID gate is the
  hard stop against entrenchment.
- No RL, no LLM labels.

## DEFERRED (2026-08-22)

r6's verified OOD sweep (WikiNews-2024 full 19.82/12.46 — beats r3
AND r5) absorbed this workstream's purpose: there is no OOD deficit
left to repair. r7 as designed costs 20h of A100-80GB for marginal
OOD gains with ID-regression risk, and its concurrent A100 footprint
is exactly what triggers Modal workspace evictions. News labeling
stopped at 5,600/13,987 windows — all committed and resumable on the
volume (label_progress.jsonl), and train_arabic_r7.py is ready with
--init-run run-006-morph. Reopen ONLY if a news-heavy client use
case emerges or Arabic OOD regresses in the wild.

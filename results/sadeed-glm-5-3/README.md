# GLM-5.3 (full) on SadeedDiac-25 (2026-09-01)

Same harness as every LLM row: neutral prompt, `temperature 0`,
structure-preserving cleanup, Misraj evaluator, 1,200/1,200 responses,
zero empty responses (41 exhausted-retry sentinels were caught by the
resume guard and re-fetched). Decode protocol: `reasoning_effort=low`
— glm-5.3 rejects `thinking: {"type": "disabled"}` outright (HTTP 400
code 1210), identical to 5.3-Flash; plain completion is inexpressible
for the 5.x generation.

## Results (Misraj evaluator, percentages)

| Protocol | DER (CE) | DER (w/o CE) | WER (CE) | WER (w/o CE) | NFDW |
|---|---|---|---|---|---|
| raw | **9.9760** | 7.9285 | 31.2581 | 25.0347 | 12.86 |
| projected zero-skip | **9.8971** | 7.8219 | 31.0748 | 24.6340 | 12.49 |

## The 5.x generation regression (attribution, same 1,200 paragraphs, convention-normalized)

| model | missing | **wrong haraqat** | extra | protocol |
|---|---|---|---|---|
| our r7 teacher | 0.11% | **2.62%** | 0.15% | greedy argmax |
| GLM-5.2 | 0.61% | **2.64%** | 0.17% | thinking disabled |
| GLM-5.3-Flash | 1.01% | **10.05%** | 0.20% | reasoning_effort=low |
| GLM-5.3 (full) | **3.75%** | **8.59%** | 0.11% | reasoning_effort=low |

Paired bootstrap (error-position deltas, 95% CI): Flash significantly
better than the full 5.3 (−2,152 positions, CI [−3,863, −546]); both
catastrophically behind GLM-5.2 (−16,251, CI [−18,028, −14,685]).
The two 5.x variants regress on different axes — Flash emits wrong
marks (10.05%), the full model leaves marks missing (3.75%) — but the
family verdict is one line: the 5.x generation lost the
classical-Arabic mark knowledge GLM-5.2 had (2.64%, matching our
dedicated 580M teacher's 2.62% to within 0.02pp). The convention
effect (U+0670) is 0.015pp here — negligible again.

## Files

- `sadeed_preds_raw.csv`, `sadeed_preds_projected.csv`

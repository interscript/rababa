# glm-4.7-flash on SadeedDiac-25 (2026-09-02)

Same harness as every LLM row: neutral prompt, `temperature 0`,
structure-preserving cleanup, Misraj evaluator, 1,200/1,200 responses,
zero empty responses. Decode protocol: `thinking: {"type": "disabled"}`
— glm-4.7-flash still accepts disabled thinking (unlike the 5.x
generation, which rejects it with HTTP 400 code 1210), so this row is
the last plain-completion GLM row.

Provider note: the endpoint sat behind sustained 429s (code 1305) —
the full set took 12 resumable passes (the #65 guard dropped
exhausted-retry empty sentinels between passes: 249 -> 208 -> 79 ->
37 -> 20 -> 13 -> 6 -> 4 -> 2 -> 0). All 1,200 final rows are real
responses.

## Results (Misraj evaluator, percentages)

| Protocol | DER (CE) | DER (w/o CE) | WER (CE) | WER (w/o CE) | NFDW |
|---|---|---|---|---|---|
| raw | **13.0035** | 10.0510 | 32.4140 | 26.7750 | 17.60 |
| projected zero-skip | **13.2256** | 10.3206 | 36.0985 | 30.1002 | 18.56 |

## Attribution (same 1,200 paragraphs, convention-normalized, rates over GT-marked positions)

| model | missing | **wrong haraqat** | extra | protocol |
|---|---|---|---|---|
| our r7 teacher | 0.11% | **2.62%** | 0.15% | greedy argmax |
| GLM-5.2 | 0.61% | **2.64%** | 0.17% | thinking disabled |
| GLM-5.3-Flash | 1.01% | **10.05%** | 0.20% | reasoning_effort=low |
| GLM-5.3 (full) | 3.75% | **8.59%** | 0.11% | reasoning_effort=low |
| glm-4.7-flash | **6.67%** | **9.01%** | 0.96% | thinking disabled |

glm-4.7-flash is the worst frontier row on DER and the only one that
regresses on BOTH axes simultaneously: the highest missing rate of
the family (6.67% — it under-diacritizes, NFDW 17.6%) alongside a
5.3-class wrong rate (9.01%). The U+0670 convention effect is
0.04pp — negligible, consistent with the family. Paired bootstrap vs
GLM-5.2 (paragraph-level simplified scorer, 10,000 resamples, seed
42): **+7.945pp, 95% CI [+7.516, +8.370], one-sided p<1e-4** —
decisively behind 5.2, whose plain-completion DER was 2.5060 raw /
2.6911 zero-skip.

The regression axis is now complete: GLM-5.2 2.51 -> GLM-5.3-Flash
8.57 -> GLM-5.3 9.98 -> glm-4.7-flash 13.00 raw DER. Every successor
generation lost classical-Arabic mark knowledge its predecessor had;
the dedicated-model thesis holds across the whole measurable family.

## Files

- `sadeed_preds_raw.csv`, `sadeed_preds_projected.csv`

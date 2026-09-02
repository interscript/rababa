# GLM-5.2 on SadeedDiac-25 (clean reproduction)

Date: 2026-08-17. Model: `glm-5.2` via z.ai API (key tier-locked out of
`glm-5.3`; 5.2 is the strongest accessible). Temperature 0, thinking
disabled (plain completion, matching the published LLM protocol), single
neutral prompt (full tashkeel incl. case endings, return only the
diacritized text). 1,200/1,200 responses; 5 initially-empty long-
paragraph responses were re-called successfully. Checkpoint:
`/tmp/sadeed_glm_glm-5_2.jsonl` (per-row, resumable); repro:
`python eval_sadeed_glm.py glm-5.2` from `interscript/rababa`.

## Results (Misraj's ArabicDiacritizationEvaluator, default protocol)

| Protocol | DER (CE) | DER (w/o CE) | WER (CE) | WER (w/o CE) |
|---|---|---|---|---|
| raw (as returned) | 2.5060 | 1.5537 | 7.9929 | 4.7509 |
| projected zero-skip | 2.6911 | 1.7179 | 8.3037 | 5.0619 |

Not-fully-diacritized words: 0.96% (raw).

## Reading

- The 2026 frontier (GLM-5.2) scores **2.51/1.55** — far from the
  published Claude-3.7 figure of 1.3941/0.7693 on the same benchmark.
  Under a neutral prompt at temperature 0, no current model we can
  test reproduces that bar.
- Our 580M r3 model (2.8429/1.7723 raw; 2.8126/1.6877 projected
  zero-skip) is within 0.3 DER of GLM-5.2 raw, and on the projected
  protocol the two **split metrics** (GLM better with case endings,
  r3 better without).

Caveats: single prompt design; the Claude-3.7 protocol details (prompt,
sampling, retries) are unpublished; GLM-5.3 untested (access denied,
HTTP 403 code 1220).

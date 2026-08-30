# GLM-5.3-Flash on SadeedDiac-25 (2026-08-31)

First protocol-matched measurement of GLM-5.3-Flash (320B total / 18B
active, native multimodal) on this benchmark. Same harness as the
GLM-5.2 reproduction (2026-08-17): neutral prompt, `temperature 0`,
structure-preserving cleanup, Misraj evaluator, 1,200/1,200 responses.

**Decode protocol delta that cannot be removed**: GLM-5.3-Flash
rejects `thinking: {"type": "disabled"}` outright (z.ai API, HTTP 400
code 1210 — "This model always engages in thinking and cannot be
disabled"); valid `reasoning_effort` values are exactly low / high /
max. This run used `reasoning_effort=low`, the closest available
analog to the GLM-5.2 run's plain completion.

## Results (Misraj evaluator, percentages)

| Protocol | DER (CE) | DER (w/o CE) | WER (CE) | WER (w/o CE) | NFDW |
|---|---|---|---|---|---|
| raw (their default) | **8.5721** | 6.5335 | 30.8406 | 24.1634 | 9.82 |
| projected zero-skip | **8.7978** | 6.6368 | 31.0323 | 24.0472 | 9.62 |

For reference, GLM-5.2 (thinking disabled): raw 2.5060/1.5537/7.9929,
zero-skip 2.6911/1.7179/8.3037. GLM-5.3-Flash is ~3.4x worse on DER
than its predecessor under the nearest equivalent protocol.

## Why the delta is real but protocol-shaped

- **Orthography**: the model emits Quranic-convention marks (dagger
  alif: عَلَىٰ, هٰذِهِ, ذَٰلِكَ; also بِهِۦ) where the benchmark's GT uses
  plain MSA forms. The evaluator skips whole sentences on word
  mismatches (survivorship in raw mode); surviving dagger-alif words
  count as diacritic errors. Not-fully-diacritized words run 9.8%.
- **Thinking floor**: reasoning cannot be turned off, so the
  plain-completion protocol the published LLM rows used is not
  expressible for this model; `low` still engaged reasoning on long
  paragraphs (reasoning_content observed in-flight).
- **Measurement hygiene**: an initial run resumed from a 140-row
  checkpoint whose rows were empty responses produced by the
  pre-fix both-knobs payload (HTTP 400s retried into empty strings) —
  Total DER read 15.96 with 11.7% catastrophic empties contaminating
  it. The 140 rows were purged and re-fetched; the numbers above are
  from a full 1,200/1,200 pass with **zero empty responses**.

## Files

- `sadeed_preds_raw.csv` — gt, model output as returned
- `sadeed_preds_projected.csv` — haraqat projected onto input letters
  (SequenceMatcher), the zero-skip protocol

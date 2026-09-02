# 05 — Head32-vs-shipped flip bootstrap (B1 extension)

Status: COMPLETE (2026-09-01, ml #120/#121 code + ml #124 record).
heb-diac: 10.985% -> 0.288% flips, delta -10.696pp, 95% CI
[-11.661, -9.726] — the repair is overwhelming. khm-latn: 2.325% ->
2.302%, CI [-0.173, +0.125] — a proper null (nothing was broken
there; head32's role is consistency). Probe-set note disclosed in
EXPERIMENTS.md. Original design below for provenance.
Status: TO BUILD. The paired-bootstrap policy covers DER deltas; the
margin flips (head32 vs shipped int8) still lack CIs because
MarginReport persists aggregates only — per-position flip vectors
are computed then discarded.

## Design

- `imf.parity.run_margin_analysis(..., dump_positions=Path)` —
  append-only JSONL of {pair_idx, pos, ref_margin, flipped} rows
  next to the margins JSON (diagnostic artifact, schema-additive)
- Rerun margin analysis for the two SHIPPED int8 zips and their
  head32 twins on the same pairs (khm + heb: one flat-ish, one the
  outlier — the informative pair), CPU on Modal, read-only w.r.t.
  artifacts
- Paired bootstrap on per-position flip indicators: is head32's
  flip-rate reduction significant per model, and is the
  confident-flip reduction (the release-relevant number) bounded
  away from zero at 95%?

## Remaining steps

- [ ] dump_positions parameter + unit test (fixture checkpoint)
- [ ] Modal runs for khm/heb shipped-vs-head32 (4 margin analyses)
- [ ] Bootstrap + record in EXPERIMENTS.md E1 (one CI line per
      artifact family) + the ledger's policy note

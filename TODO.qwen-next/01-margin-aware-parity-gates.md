# 01 — Margin-aware parity gates (adopt now)

Source: Qwen3.8-Flash-Next quantization practice (Unsloth day-0 analysis) —
the dense backbone tolerates aggressive quantization, but random-access
tensors (embedding tables, output heads, PLE lookup tables) degrade fast
below 4-bit. Precision floor is a property of **how a tensor is read**,
not just its size. They validate with KL-divergence on logits, not
output equality alone.

## Gap in our stack

`ml-models/src/imf/parity.py` gates releases on CER-delta between the
torch reference and the ONNX zip (0.2pp fp32 → 3.0pp int4). Our byte
students have **flat top-1 margins** (the greedy-vs-beam lesson: tha-small
published 12.06 PER was really 2.85). On flat distributions, quantization
noise below exact-match detection can still flip near-tie argmaxes — a
golden-set CER gate can pass while decode fragility ships.

## Deliverables

1. `run_margin_analysis(model, zip_path, pairs)` in `src/imf/parity.py`:
   teacher-forced forward on both sides (torch decoder vs ONNX decoder
   session), per-token top1−top2 logit margins, argmax flip rate, KLD.
2. `MarginReport` dataclass + JSON serialization; written next to the zip
   as `<id>-margins-<precision>.json` (diagnostic, not a release blocker
   yet — schema churn avoided).
3. Unit test on the fixture checkpoint (`make_fixture_checkpoint`).
4. Real validation: khm-latn-1.0-fp16.zip (local) + one int8/int4 zip.
5. Policy (documented in docs/EXPERIMENTS.md): embeddings + lm_head +
   any memory tables are a separate quantization class — stay fp16 (≥int8
   at minimum) when the body is quantized.

## Protocol (paper-ready)

- Probe set: the parity golden inputs (same pairs the CER gate uses).
- Metrics: flip_rate = fraction of teacher-forced positions where
  argmax(torch) ≠ argmax(onnx); margin quantiles (p1/p10/p50) of the
  reference; KLD mean; per-precision.
- Hypothesis: flip_rate correlates with cer_delta but detects fragility
  earlier; int4 artifacts show near-tie flips invisible to the CER gate.

## Status

- [x] run_margin_analysis + MarginReport in parity.py
- [x] Wired into modal_export parity gate (every export emits margins)
      + read-only `margins` entrypoint for published zips
- [x] Unit test (fixture checkpoint, synthetic near-tie logits) — 8 pass
- [x] **Full-catalog table measured (12 rows, docs/EXPERIMENTS.md E1)**:
      fp32 exact everywhere; fp16 benign except khm 2.93% (flattest
      margins, p50 0.121); int8 0.26–9.34%. **heb-diac-1.1 int8 = the
      outlier: 9.34% flips, only 20% near-tie** — 80% of argmax flips at
      confident positions, invisible to the CER gate. Open item:
      re-examine the Hebrew int8 artifact (per-channel quantization or
      serve fp16) + add a margin threshold to the release policy.
      tha-g2p-small int4: 0.26%, all near-tie — flat-but-consistent.
- [x] **Root cause found + export default fixed (fcbe5d3)**: the outlier
      was the quantized *head*. Probes on the same pairs: per-channel
      alone 8.50% (+25% size, rejected); **head-fp32 body-int8 → 0.26%
      flips (36×), KLD 47× lower, 100% near-tie, +0.4% size**.
      `export_zips` now excludes `/lm_head/MatMul` from int8 by default.
      Shipped int8 zips predate it; re-export = release decision (open).
- [x] Policy registered in docs/EXPERIMENTS.md (E1)
- [x] **Corrected-artifact sweep (2026-08-29, authorized)**:
      `rebuild_int8_head32` (PR #76) rebuilds every shipped int8 zip
      with the head in fp32 and gates it (parity in-zip + margin
      budget). Artifacts land as {mid}-int8-head32.zip; swap-in is a
      version decision pending results. Sweep running across khm/urd×2/
      heb/tha; tha int4 left as-is (benign: 0.26%, all near-tie).

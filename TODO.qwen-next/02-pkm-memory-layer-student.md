# 02 — PKM memory-layer student (client-tier v2, the big one)

Sources:
- arXiv 2601.21204 "Scaling Embeddings Outperforms Scaling Experts"
  (LongCat-Flash-Lite: 68.5B params, >30B in embedding/lookup tables,
  ~3B active — embedding scaling beats expert scaling on the Pareto
  frontier in specific regimes).
- Qwen3.8-Flash-Next: first public model on that recipe (51B N-gram/PLE
  tables, ~6B active).
- Precedent at our modality: product-key memory was demonstrated on
  **character-level** LM (Lample et al., NeurIPS 2019, enwik8/text8).

## Why this is the right experiment for us

Wrong turn 1 measured: sub-100M from-scratch byte students collapse; a
pretrained backbone is non-negotiable; the client tier is pinned at
ByT5-small 300M → 246 MiB int8 (Arabic student: 8.26 DER full-set vs
teacher 2.58 — a 5.68pp gap).

The LongCat result reframes the axis: **parameters and compute are
separable**. Keep compute at ByT5-small, add lookup capacity. Haraqat
restoration is heavily lexical knowledge (word identity + local context)
— table-friendly, not compute-friendly. It is also the architectural
version of our r6/r8 aux-task finding: tagged knowledge injection works;
a memory layer is where that knowledge can physically live.

## Design (as launched)

- Backbone: `google/byt5-small` (pretrained, per the non-negotiable rule).
  Measured at launch: 300M base, d_model 1472, **4 decoder blocks**
  (ByT5 depth lives in the encoder) — memory layers on decoder blocks
  [-1, -2, -3].
- Memory: product-key factorization, C=128 keys/half → 16,384 slots,
  top-k=32 sparse reads → **+85.9M params (+29%)** at near-zero FLOPs.
  Zero-init output gate preserves the pretrained function at step 0
  (unit-tested: logits bit-identical pre/post injection).
- Training: identical to ara-diac-small run-002 — teacher r6 frozen,
  same r5-units corpus (29,322 pairs), same `teacher_labels_v2.jsonl`
  (copied into the run dir; labels trusted complete — no relabeling),
  same seed → **single-variable comparison**.

## Protocol + gate (pre-agreed)

- Spec `ara-diac-small-pkm` in ml-models/src/gpu/modal_distill.py.
- Gate: windowed zero-skip Misraj DER-CE, full 1,200 paragraphs,
  ≤ 3.07 (the run-002 gate) — and the comparison target is run-002's
  8.259 full-set student / 2.5815 teacher.
- Verdict rule: PKM wins if it closes ≥1.0pp of the 5.68pp gap at equal
  decode-time compute (memory reads are gathers, not matmuls). If no
  movement, the capacity story is wrong — the gap is modeling/optimization,
  publishable as a negative either way.
- Onnx export (opset-14 TopK/Gather) only if it wins; noted, not built yet.

## Launch

- A10G distill slot (never competes with A100 teacher runs) — fits under
  the ≤2-big-GPU-apps budget alongside r8. 10,995 steps, ~7h.
- Detached; chain watchdog crash-relaunches, then runs
  evaluate_der, then launches the Muon A/B arm (TODO 03), then evals it.

## Status

- [x] pkm.py (PKMLayer + byt5 injection helper) + smoke tests (6 pass)
- [x] Spec wiring in modal_distill.py (train + eval load paths)
- [x] Launched detached: rababa_arabic_distill_small/run-003-pkm
- [x] Registered in ml-models docs/EXPERIMENTS.md (E2)
- [x] Engagement probe (`pkm_gates`): step-500 gates 0.0008/0.0028/0.0019
      — all three off zero; memory branch engaged (CE 2.07 → 0.086 @ 900)
- [x] **VERDICT (2026-08-28): 7.5553 full-set DER vs run-002's 8.259 —
      0.704pp of the 5.677pp gap closed (12.4% relative), below the
      pre-registered ≥1.0pp win bar.** Positive direction, honestly
      reported; capacity is real but not the dominant term. Gate
      engagement verified (gates off zero, CE 2.07→0.02). Muon arm (E3)
      now tests the optimization half of the remaining gap.

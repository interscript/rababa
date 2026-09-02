# 03 — Muon optimizer A/B (one cheap test, low expected gain)

Source: Qwen3.8-Flash-Next / LongCat-Flash-Lite trained with Muon
(orthogonalized momentum via Newton–Schulz); "~1/9th training cost" claims
circulate for from-scratch pretraining.

Our prior: fine-tunes are **knowledge-limited, not optimization-limited**
(RL was a measured negative; data/supervision-side levers won). So the
expected gain on a 3-epoch distill is small — but the test is cheap and
runs in the same A10G slot after the PKM run.

## Design

- `muon.py` in ml-models/src/gpu/: single-file Muon — Newton–Schulz
  orthogonalization applied to 2D weight matrices (hidden/FFN/attn
  projections); embeddings, layer norms, and the lm_head stay on AdamW
  (standard split; embeddings are not orthogonalizable objects).
- Flag `optimizer: muon|adamw` in the distill spec (default adamw —
  nothing changes for existing specs).
- A/B: same spec as TODO 02 (`ara-diac-small-pkm`), same seed 42, only
  the optimizer differs → run-003-pkm (adamw) vs run-004-pkm-muon.

## Protocol + gate

- Metric: windowed DER-CE on the val slice during training + full
  1,200-paragraph Misraj at the end; wall-clock per step recorded.
- Adopt if: ≥0.3pp DER improvement at equal steps AND no stability
  regressions (loss spikes, grad-norm blowups) AND step overhead <15%.
- Either direction is paper-reportable (training-methods appendix):
  "Muon vs AdamW on byte-level distillation fine-tunes" is unmeasured
  territory for seq2seq students.

## Status

- [x] muon.py (Newton–Schulz, param-group split; shared.weight routing
      fixed for transformers 5.x) + unit tests
- [x] Spec `ara-diac-small-pkm-muon` (run-004) wired in modal_distill.py
- [x] **VERDICT (2026-08-28): 4.8287 vs 7.5553 — −2.727pp from the
      optimizer alone; adopt gate (≥0.3pp) exceeded 9×. ADOPTED.**
      Training CE ~0.007 vs ~0.02 at equal steps; ~1.2s/step vs ~3.4s;
      no stability events. Gap decomposition: ≈0.70pp capacity +
      2.73pp optimization + 2.25pp residual.
- [x] Factorial cell 4 (vanilla+Muon, run-005-muon) landed 2026-08-28:
      **5.2945** — the 2×2 closes cleanly (optimizer alone −2.96pp,
      memory alone −0.70pp / −0.47 under Muon, combined −3.43pp).
      Paper carries the full factorial table.

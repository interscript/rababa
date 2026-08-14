# Phase 2 — rababa Hebrew (Tier 1: direct supervised)

## Goal
Cut `rababa_hebrew-v0.1.0`. Mirror Phase 1 architecture with Hebrew
vocab and data. No teacher.

## Tasks

### 2.1 Dataset acquisition
- **Dicta Nakdan API** (open): `https://nakdan.dicta.org.il/api`. Returns
  nikudized Hebrew for input text. Use as silver labels.
- **Hebrew NC**: ~5 K manually nikudized sentences from the Dicta project.
- **Hebrew Wikisource** nikudized poetry/prose: ~10 K examples.
- Combine → ~15 K gold + ~50 K silver (Dicta predictions on raw NC).
- Filter: keep only predictions where Dicta's confidence ≥ 0.95 OR
  in Wikisource gold set.

### 2.2 Vocab
Hebrew nikud marks (~30): kamatz, patach, segol, hiriq, cholam, dagesh,
shin-dot, sin-dot, etc. Vocab size ~40 (vs 16 for Arabic).

`rababa/src/rababa/datasets.py` — add `fetch_hebrew_dataset()`.

### 2.3 Tier 1 student training
- Same 6-layer char transformer (~25 M params).
- Compute: 1× A100 40 GB, 5 epochs, ~2 h (smaller dataset than Arabic).
- Acceptance for v0.1.0: **DER ≤ 20%** on Dicta gold test split (research baseline).

### 2.4 ONNX export
- Same fixed shape `[batch=32, max_len=200]`.
- int8 quantization.
- Acceptance: ≤ 25 MB int8.

### 2.5 Release
- Cut `rababa_hebrew-v0.1.0` tag.
- Update manifest: `rababa_hebrew` version `0.1.0`, status `research`.
- TS: `setRababaConfig("hebrew-v0.1", { model: "...", config: {...} })`.
- Ruby: `Interscript.rababa_configs["hebrew-v0.1"]`.

## Acceptance
- [ ] DER ≤ 20% on Dicta gold test
- [ ] int8 ONNX ≤ 25 MB
- [ ] TS parity 100% on `var-heb-Hebr-Hebr-nikud` test vectors (if map exists)

## Open questions
1. **Is there an existing map that uses Hebrew nikud?** Search maps repo for `var-heb-Hebr` or similar.
2. **Data licensing**: verify Dicta NC is redistributable for ONNX model training. If not, keep data local; ship only ONNX.

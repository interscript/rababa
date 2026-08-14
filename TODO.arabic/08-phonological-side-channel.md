# 08 — Phonological side-channel (iltiqā' as-sākinayn)

## Why
The iltiqā' as-sākinayn rule forbids two consecutive sukun (no-vowel)
positions in Arabic phonology. We already detect this in
`scripts/clean_tashkeela_sadeed.py::resolve_iltiqaa_as_sakinayn`.
Currently it's a CLEANER (modifies input). We can also expose it as
an INPUT FEATURE — a binary "this position violates iltiqā'" mask
fed alongside the char IDs.

This gives the model free phonological signal without changing the
arch.

## Tasks

### 8.1 Feature extractor (`src/rababa/features.py`)
- `compute_phonological_features(text: str) -> list[dict]` per position.
- Initial feature set: `iltiqaa_violation`, `word_initial`, `word_final`.
- Future: `consonant_class` (moon/sun), `vowel_length`.

### 8.2 ModernCharTransformer: feature embedding
- Add `feature_dim` constructor param.
- New `nn.Embedding(feature_vocab_size, dim)` added to char embedding.
- When `feature_dim=0` (default), no change — backward compatible.

### 8.3 Wire into datasets + collate
- Dataset returns `(input_ids, target_ids, feature_ids)`.
- Collate pads feature_ids alongside src.
- When features disabled, skip.

### 8.4 Config flag
- `cfg.model.features: ["iltiqaa", "word_boundary"]` (default empty list).

## Acceptance
- [ ] `compute_phonological_features("الْعَرَبِيَّة")` returns the
      expected mask.
- [ ] Training with features enabled reduces DER vs baseline by ≥ 1%.
- [ ] Features disabled by default → existing configs unchanged.

## Files
- `src/rababa/features.py` (new)
- `src/rababa/models/modern.py` (add feature embedding)
- `src/rababa/datasets.py` (add feature extraction to loaders)
- `src/rababa/training/collate.py` (collate features)
- `tests/test_features.py` (new)

## Open questions
- Feature ablation: which single feature carries the most signal?
  Run a small sweep before default-enabling.

# 13 — Spec coverage

## Why
New modules (trie inference, multi-seed, ELECTRA, noisy student,
LatentMoE, engram) need specs. Existing modules also have gaps.

## Standards (per global CLAUDE.md)
- **No doubles.** Use real model instances or `Struct.new` for plain data.
- **Test behavior, not implementation.** Assert on output and state,
  not "should have_received".
- **Spec real edge cases.** Empty input, single-token, max-len,
  unicode normalization, PAD/BOS/EOS interaction.

## Tasks

### 13.1 New module specs
- `tests/decoding/test_constrained.py` — trie decode happy path + OOV fallback
- `tests/training/test_distill.py` — distillation loss + α schedule
- `tests/training/test_noisy_student.py` — labeling + augmentation
- `tests/training/test_electra.py` — generator + discriminator losses
- `tests/models/test_moe.py` — router + balance loss
- `tests/models/test_engram.py` — write/read + cosine retrieval

### 13.2 Existing module gap-fill
- `tests/test_pretrain.py` — add MLM smoke test on tiny corpus
- `tests/training/test_supervised.py` — multi-head + single-head coverage
- `tests/training/test_resume.py` — checkpoint save/load roundtrip
- `tests/test_features.py` — phonological feature extractor

### 13.3 CI gate
- `pytest --cov=rababa --cov-fail-under=80` in CI.
- New code MUST land with specs in same PR.

## Acceptance
- [ ] Coverage on `src/rababa/` ≥ 80%.
- [ ] All new modules have at least 1 happy-path + 1 edge-case spec.

## Files
- `tests/**` (multiple new files)
- `.github/workflows/ci.yml` (add coverage gate)

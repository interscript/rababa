# 12 — Bigger Arabic corpus

## Why
Current corpus: GPLv2 Tashkeela-full + Sadeed HF + QCRI EMNLP 2025
= ~2.1M lines, ~80M words. SOTA Arabic diacritizers use 5-10× this.
Sources to add:
- **WikiDiplomatic** (if it exists for Arabic) — official Arabic text.
- **OpenITI** — pre-modern Arabic corpus (~5K books, fully diacritized
  in many places).
- **CC-100 Arabic filtered** — common-crawl Arabic with auto-diacritization.
- **Tashkeela+ (Hu et al. 2024)** — refined superset of Tashkeela.

## Tasks

### 12.1 Corpus registry (`src/rababa/data_sources/`)
- Each source is its own module: `openiti.py`, `cc100_arabic.py`, etc.
- All conform to `CorpusSource` protocol: `fetch() -> Path`, `clean(text) -> str`.
- This is OCP-compliant: new corpus = new file, no edits to existing.

### 12.2 Combined corpus builder
- Extend `_build_arabic_combined_corpus` in `modal_app.py` to pull from
  all registered sources.
- Skip missing sources gracefully (same pattern as Sadeed HF).

### 12.3 Provenance tracking
- For each line in the combined corpus, record source.
- `/datasets/arabic-combined/provenance.jsonl` per line.

## Acceptance
- [ ] At least one new corpus added; combined size grows by ≥ 1M lines.
- [ ] DER on test split improves after retraining on bigger corpus.

## Files
- `src/rababa/data_sources/__init__.py` (new — registry)
- `src/rababa/data_sources/openiti.py` (new)
- `src/rababa/data_sources/cc100_arabic.py` (new)
- `modal_app.py` (extend corpus builder)

## Open questions
- License compatibility: OpenITI is mixed. We need to verify per-source
  license terms and only ship models trained on compatible data.

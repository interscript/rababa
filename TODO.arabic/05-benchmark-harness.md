# 05 — Benchmark harness (Fadel + SadeedDiac-25)

## Why
Without a standard benchmark, every "DER improved" claim is anecdotal.
We need:
- **Fadel et al. test split**: classic 2014 benchmark, comparable to
  literature.
- **SadeedDiac-25** (QCRI EMNLP 2025): modern refined test set, more
  challenging, less dup-with-train contamination.
- **Word-level / sentence-level DER**: both per-position and
  per-sentence-exact-match.

## Tasks

### 5.1 Bundled benchmarks
- `test-datasets/benchmarks/fadel/{train,val,test}.txt` (already
  partially present — verify format).
- `test-datasets/benchmarks/sadeed-diac-25/{...}` (clone from QCRI
  repo on image build).

### 5.2 Unified evaluator (`src/rababa/benchmark.py`)
- Replaces ad-hoc eval calls.
- `run_benchmark(model, task, split, dataset_name) -> BenchmarkResult`.
- Returns: der, wer, per-haraqat-class der, per-genre der, n_examples.

### 5.3 Modal entrypoint (`modal_app.py::benchmark`)
- `benchmark --task rababa_arabic_pro --datasets fadel,sadeed-diac-25`
- Loads best.pt, runs evaluator, writes JSON to `/models/{task}/benchmark-{version}.json`.

### 5.4 Regression baseline
- Pin baseline DER per dataset per version.
- CI job (post-PR) re-runs benchmark and fails if DER regresses > 0.5%.

## Acceptance
- [ ] `benchmark --datasets fadel` produces DER on Fadel test split.
- [ ] `benchmark --datasets sadeed-diac-25` produces DER on Sadeed.
- [ ] Baseline JSON committed at `tests/baselines/benchmark-v0.1.0.json`.

## Files
- `src/rababa/benchmark.py` (new)
- `modal_app.py` (add `benchmark` function)
- `scripts/run_benchmark.py` (new — CLI wrapper)
- `test-datasets/benchmarks/` (data)
- `tests/test_benchmark.py` (new)

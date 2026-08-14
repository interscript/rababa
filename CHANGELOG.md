# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — 2026 cross-model techniques (v0.7.0 in progress)

Survey of 2026 ML techniques beyond DS-V4 / Kimi-K3 / Qwen 3.8 stack.
Full analysis in `docs/CROSS_MODEL_2026_analysis.md`. Implemented:

**Architectural**:
- **ResFormer** (arXiv:2410.17897, ACL 2025): value residual
  `V_n = λ_1·V_1 + λ_2·V_n` before attention. Sparse mode (last N layers,
  λ_1=5.0) per paper Table 3. In `models/modern.py`.

**Optimizer (Muon variants)** — all in `training/optim.py`:
- **Spectral Cap Muon** (2026): Frobenius-norm cap on orthogonalized updates.
- **HTMuon** (arXiv:2603.10067, ACL 2026): heavy-tail α-blend with raw momentum.
- **AdaMuon** (arXiv:2507.11005): element-wise second-moment estimator.
- **NorMuon** (arXiv:2510.05491): neuron-wise adaptive scaling.

**Training**:
- **Early stopping** (`cfg.train.early_stopping_patience`): breaks if val_loss
  doesn't improve for N epochs. Prevents overfitting drift on small datasets.
- **MetricsLogger wired into secryst supervised** (was only in CTC path).

**Configs** (rababa):
- `rababa_hebrew_dsv4.yaml` — DS-V4 Tier 1 only (from v0.6.x).
- `rababa_arabic_pro_dsv4.yaml` — same, Arabic Pro.
- `rababa_hebrew_resformer.yaml` — full stack (DS-V4 + ResFormer + 4 Muon variants).
- `rababa_arabic_pro_resformer.yaml` — same, Arabic Pro.
- `rababa_hebrew_resformer_only.yaml` — ablation: ResFormer without DS-V4.
- `rababa_hebrew_adamuon.yaml` — ablation: AdaMuon+NorMuon only (no architectural changes).
- `rababa_hebrew_resformer_reg.yaml` — stronger regularization (dropout 0.3, wd 0.05).
- `rababa_arabic_pro_pretrain_resformer.yaml` — pretrain variant (where techniques should help).

**Configs** (secryst):
- `secryst_thai_ipa_resformer.yaml` — full stack for Thai→IPA.

**Scripts**:
- `scripts/compare_techniques.py` — N-way A/B comparison.
- `scripts/auto_compare.py` — auto-pull metrics + compare.
- `scripts/inspect_resformer_lambdas.py` — extract learned λ from checkpoint.

**Empirical findings (Hebrew, 29K supervised pairs)**:
| Variant | Best val_loss | vs baseline |
|---|---|---|
| Baseline v0.6.0 | **3.36** | — |
| AdaMuon+NorMuon | 4.69 | +40% (closest, most stable σ=0.04) |
| ResFormer Reg | 5.11 | +52% |
| ResFormer | 6.11 | +82% |
| DS-V4 Tier 1 | 6.30 | +88% (most stable but worst) |

**Conclusion**: Architectural techniques (DS-V4, ResFormer) consistently hurt
Hebrew supervised by adding capacity the small dataset can't support.
Optimizer-side techniques (AdaMuon+NorMuon) are the most promising direction
for small-data supervised. Pretraining (Arabic 75M words) is where the
architectural techniques should actually help — runs in flight.

### Added — Modern training pipeline (`src/rababa/`)

Modern reimplementation of rababa with Modal-native training and
browser-deployable ONNX models. Coexists with the legacy 2021 CBHG
code in `python/`; new work happens here.

- **`src/rababa/`** — modern Python package (PEP 621, hatchling).
  - `config.py` — OmegaConf loader: `base.yaml` + `<task>.yaml`.
  - `constants.py` — Arabic alphabet + haraqat Unicode codepoints.
    Ported from `python/arabic/util/constants.py` so encoder IDs
    match the 2021 trained model exactly (legacy baseline is
    directly comparable).
  - `encoder.py` — `ArabicEncoder` (text → token IDs).
  - `datasets.py` — `TashkeelaDataset` (parallel input/target pairs)
    + `ArabicMLMDataset` (raw text → BERT-style masked examples).
  - `models/student.py` — `CharTransformer`: 6-layer encoder, 384
    dim, 6 heads, ~11M params. Sized for browser deployment (~3 MB
    after int8).
  - `models/mlm.py` — `MLMHead` + `MLMModel` wrapping the student
    for char-level MLM pretraining. Tied input/output embeddings.
  - `training/supervised.py` — Tier 1 training loop with AMP,
    cosine schedule, grad-clip. Accepts `init_from_pretrain` to
    load an MLM-pretrained encoder.
  - `training/pretrain.py` — MLM pretraining loop + collate.
  - `training/collate.py` — padding + truncation to `max_len=200`.
  - `export.py` — PyTorch → ONNX (fixed shape) + int8 quantization.
  - `evaluate.py` — DER + per-example accuracy.
  - `benchmark.py` — ONNX-vs-test-split harness. Produces the JSON
    used to verify "must not regress" before shipping.
  - `cli.py` — `rababa-pretrain` / `rababa-train` /
    `rababa-export` / `rababa-evaluate` entry points.

- **`modal_app.py`** — Modal definitions:
  - `fetch_data` — verify Tashkeela splits present on volume.
  - `pretrain` — A100, ~6h, MLM char-level pretraining.
  - `train` — A100, ~3h, Tier 1 supervised fine-tune. Accepts
    `--init-from-pretrain` to consume a pretrain checkpoint.
  - `export_onnx` — A10G, ONNX fp32 + int8.
  - `evaluate` — A10G, DER + accuracy on test split.

- **`configs/`** — `base.yaml` + `rababa_arabic.yaml` +
  `rababa_arabic_pretrain.yaml`.

- **`tests/`** — 23 tests covering config, encoder, dataset, model,
  supervised training, MLM pretraining, ONNX export, int8
  quantization. CPU-runnable; `pytest tests/`.

- **`TODO.modernize/`** — phased plan:
  - `00-plan.md` — overview.
  - `01-phase0-foundations.md` — framework (done).
  - `02-phase1-rababa-arabic.md` — Tier 1 supervised (pending).
  - `02a-mlm-pretrain.md` — architectural decision: char-level MLM
    pretraining as the SOTA-2026 upgrade path. Documents rejected
    alternatives (MARBERT init, Sadeed distillation).
  - `03-phase2-rababa-hebrew.md` — Hebrew (pending).
  - `04-training-and-benchmark.md` — full Arabic + Hebrew training
    + benchmark protocol.
  - `05-blog-post-outline.md` — outline for the announcement post.
  - `06-phase5-production.md`, `07-phase6-maintain.md`.

### Architecture — MLM char-level pretraining (Phase 0.5)

Reviewed 2024–2026 SOTA (Sadeed 1.5B decoder-only, SUKOUN BERT,
PTCAD, CATT, AyutthayaAlpha) and chose char-level MLM pretraining
as the architectural upgrade. Rationale (full doc in
`TODO.modernize/02a-mlm-pretrain.md`):

- Same `CharTransformer` architecture — no browser-deployment change.
- No WordPiece tokenization mismatch (the killer for MARBERT init).
- Fits Modal budget (~6h pretrain + ~3h fine-tune on A100).
- No HF weight dependency — we pretrain from scratch on raw Arabic.

### Benchmark

Established baseline by running `models-data/arabic-model.onnx`
(2021 CBHG, 60 MB fp32) against the Tashkeela test split (2,496
examples) via the new `benchmark.py` harness:

| Metric                | Legacy 2021 |
|-----------------------|-------------|
| DER                   | **4.52%**   |
| Per-example accuracy  | 8.85%       |
| Model size            | 60 MB       |

Result file: `benchmark-legacy-arabic.json`.

**v0.1.0 acceptance: new model DER must be ≤ 4.52%** (parity) and
ideally ≤ 4.0% (clear improvement). The earlier "≤ 15%" target in
`02-phase1-rababa-arabic.md` was set before benchmarking the legacy
model — the real bar is much higher.

## [Latest]

See GitHub releases for detailed release notes: https://github.com/interscript/rababa/releases

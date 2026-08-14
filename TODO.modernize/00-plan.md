# rababa-v3 / secryst-v2 — Modernize with Modal

## Decisions (confirmed)

1. **Secryst is a standalone repo** at `interscript/secryst/` (mirroring `rababa/`).
2. **int8 only** — no int4 variant.
3. **No teacher in Tier 1** — direct supervised training on gold data. Distillation is Tier 2 (only if DER isn't good enough). Teacher pretraining advantage is implicit in the character transformer's architecture (modern attention + regularization beats the 2021 CBHG without needing pretraining).
4. **Iterative release cadence** — `v0.1.0 → v0.5.0 → v1.0.0`, never jump straight to v1.0.0.
5. **Paid Modal** — parallel training runs OK.

## Constraints
- **One repo per model**: rababa stays in `rababa/`, secryst in `secryst/`. Shared training infra lives in `ml-models/`.
- **Manifest versioning is the API**. Runners look up by task key. `version` is semver.
- **Browser budget**: ≤ 30 MB int8 per model. Server budget: ≤ 200 MB fp16 per model.
- **Zero breaking changes** during the transition: old model URLs continue to serve while new ones roll out behind a version flag.

## Current state (audit)

| | rababa | secryst |
|---|---|---|
| 2021 model | ✅ CBHG, 60 MB fp32 | ❌ no model |
| Training | Python in `rababa/python/`, local GPU | not started |
| Ruby gem | `rababa/lib/rababa/arabic.rb` (OnnxRuntime) | `lcs/` has docs only; no Ruby code |
| TS runtime | `interscript-ts/src/ml/models/rababa/` (works, 100% on 1 test) | framework only; no implementation |
| Manifest | `rababa_arabic@0.0.0` preview | `secryst_thai_ipa@0.0.0` preview |
| Data | Tashkeela++ (not yet fetched) | Wiktionary Thai-IPA (not yet fetched) |

## Architecture (after)

```
rababa/                       (existing repo, modernized)
├── pyproject.toml            # PEP 621
├── modal_app.py              # Modal definitions — train, export, eval
├── src/rababa/
│   ├── datasets.py           # Tashkeela++, Hebrew NC
│   ├── models/
│   │   ├── student.py        # 6-layer char transformer (Tier 1)
│   │   └── quantize.py       # ONNX → int8 with calibration
│   ├── training/
│   │   ├── supervised.py     # Tier 1: direct training on gold labels
│   │   └── distill.py        # Tier 2: teacher-as-noisy-oracle on UNLABELED data
│   ├── export.py             # PyTorch → ONNX, fixed shape verified
│   └── evaluate.py           # DER / PER on gold test split
├── tests/
├── models/
├── configs/
├── Dockerfile                # local GPU fallback
└── README.md

secryst/                      (NEW standalone repo, mirrors rababa layout)
├── pyproject.toml
├── modal_app.py
├── src/secryst/...           # Thai → IPA focus
├── models/
├── configs/
└── tests/

ml-models/                    (SHARED infra, unchanged)
├── src/framework/            # config, registry, trainer, exporter
├── src/tasks/                # task configs + data modules
├── modal_app.py              # dispatch — calls into rababa/secryst
└── tests/
```

## Teacher reconsidered

**Original plan**: ByT5-base teacher → distill into compact student.
**Problem**: teacher brings dataset bias; doesn't "know rules" beyond what's in the gold data.
**New plan**:

- **Tier 1 (default)**: Direct supervised training on gold labels. ~25M char transformer with modern attention + regularization. This matches the existing `ml-models/src/tasks/rababa_arabic/config.yaml`.
- **Tier 2 (only if Tier 1 isn't good enough)**: Use a teacher as a **noisy augmentation oracle**, not as the source of truth:
  - Teacher is a larger model trained on the same gold data.
  - Teacher labels large amounts of UNLABELED Arabic text.
  - Student trains on `gold_data ∪ teacher_labeled_data`.
  - Student's labels still come from gold data; teacher-labeled data is augmentation.
  - Acceptance: Tier 2 only if student DER on gold test improves by > 2 absolute points.

This means we **don't burn a teacher GPU until we know we need one.**

## Phases

### Phase 0 — Foundations (1 week)
- Modal auth (`modal token new`), volumes (datasets, checkpoints, models).
- Dataset fetch pipelines: Tashkeela++, Hebrew NC, Wiktionary Thai-IPA.
- Shared base config in `ml-models/configs/base.yaml`.
- `modal_app.py` skeleton in both `rababa/` and `secryst/`.
- See [01-phase0-foundations.md](01-phase0-foundations.md)

### Phase 1 — rababa Arabic (2 weeks)
- Tier 1: direct supervised training on Tashkeela++. 6-layer char transformer.
- int8 ONNX export.
- Cut `rababa_arabic-v0.1.0` (research quality, DER baseline).
- See [02-phase1-rababa-arabic.md](02-phase1-rababa-arabic.md)

### Phase 2 — rababa Hebrew (1 week, parallel with Phase 1)
- Same architecture, Hebrew vocab + Dicta/NC data.
- Cut `rababa_hebrew-v0.1.0`.
- See [03-phase2-rababa-hebrew.md](03-phase2-rababa-hebrew.md)

### Phase 3 — secryst Thai-IPA from scratch (2 weeks)
- Build standalone `secryst/` repo.
- Wiktionary Thai-IPA dataset fetch + augmentation.
- Tier 1 student training.
- Cut `secryst_thai_ipa-v0.1.0`.
- See [secryst/TODO.modernize/04-phase3-secryst-thai-ipa.md](../../secryst/TODO.modernize/04-phase3-secryst-thai-ipa.md)

### Phase 4 — Wire secryst into TS + Ruby (1 week)
- TS: `src/ml/models/secryst/`, `secryst()` stdlib function, interpreter async dispatch.
- Ruby: `SecrystAdapter`.
- End-to-end test.
- See [secryst/TODO.modernize/05-phase4-secryst-wiring.md](../../secryst/TODO.modernize/05-phase4-secryst-wiring.md)

### Phase 5 — Production deployment (1 week)
- CDN, caching, server fallback, A/B rollout, CI release workflow.
- Cut `v1.0.0` once each model passes 95% test pass rate over 1 month.
- See [06-phase5-production.md](06-phase5-production.md)

### Phase 6 — Maintain (ongoing)
- Quarterly retrain, SOTA tracker, new task onboarding.
- See [07-phase6-maintain.md](07-phase6-maintain.md)

## Release cadence

| Version | Meaning | Quality bar |
|---|---|---|
| `v0.1.0` | First research release | Tier 1 student trained. DER baseline measured. Not deployed. |
| `v0.5.0` | Improved architecture | DER improved by > 3 absolute points. Internal deployment OK. |
| `v1.0.0` | Stable release | ≥ 95% test pass rate. ≤ 30 MB int8. Deployed to website. |
| `v1.X.0` | Architecture iteration | New teacher / new student / new data. Manual review. |
| `v1.0.X` | Patch (re-train) | Quarterly refresh. CI green = auto-merge. |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Modal outage during training | Resume from checkpoint on volume. Local Docker fallback for CPU smoke. |
| Tier 1 student under-fits | Trigger Tier 2 distillation. Teacher-as-oracle on unlabeled data only. |
| New model regresses on test vectors | Old version remains in manifest. Rollback = bump manifest. |
| Tashkeela++ license changes | Vendor dataset with provenance; ONNX export doesn't depend on dataset availability. |
| ONNX export shape mismatch with TS runtime | Fixed shape enforced in `export.py`; CI parity test on 100 examples. |
| Browser memory budget exceeded | Student ≤ 25 M params; int8; ≤ 25 MB. |
| Secryst has no Ruby code today | Implement Ruby adapter as part of Phase 4. |

## Open questions

1. **Modal compute sizing**: A100 80 GB vs A10G 24 GB. A100 for big runs, A10G for development? Or standardize on A100?
2. **Telemetry vendor**: roll our own (Cloudflare Worker + Logflare), or use PostHog / Plausible? Recommend: keep dead simple.
3. **A/B cohort sampling**: cookie-based or session-based? Session-based for privacy.
4. **Quarterly retrain cost estimate**: A100 × 3 tasks × 5 hours each × 90 days = ~$1.5K/year at Modal rates. Confirm budget.

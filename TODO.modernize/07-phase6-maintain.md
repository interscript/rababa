# Phase 6 — Maintain / improve

## Goal
Keep the models fresh, accurate, and aligned with SOTA without
becoming a maintenance burden.

## Tasks

### 6.1 Quarterly retrain

`ml-models/modal_app.py` — scheduled job (Modal cron):
```python
@app.function(schedule=modal.Period(days=90))
def quarterly_retrain():
    for task in ["rababa_arabic", "rababa_hebrew", "secryst_thai_ipa"]:
        fetch_data(task)
        train_student(task)
        # Bump to "0.0.X" (next patch) — never auto-bump minor/major
        export_onnx(task, version="0.0.X")
    # Open PR with metrics report
    open_pr_with_metrics(...)
```

Policy:
- Patch bumps: zero review needed (CI green = auto-merge).
- Minor bumps: manual review of DER/CER deltas vs last minor.
- Major bumps: full review + on-call notification.

### 6.2 SOTA tracker

Watch-list (quarterly review):
- **ByT5** (current teacher). Successors: ByT5 v2 (?), mBART, NLLB-200 distilled.
- **Char transformer** (current student). Successors: Mamba-2 SSM (linear time, good for long Thai compounds), RWKV.
- **Quantization**. GPTQ, AWQ, SmoothQuant for int4.

When a successor demonstrates > 1.5 DER/CER absolute improvement on
test splits, kick off a new Phase 1 (rababa) or Phase 3 (secryst).

### 6.3 New task additions

When a new map needs ML:
1. Add `src/tasks/<new_task>/` with config + data + student.
2. Modal app picks it up automatically.
3. Train, eval, release as v1.0.0 with default OFF in manifest.
4. Graduate to `status: stable` after 1 month at >95% test pass rate.

Candidates:
- `khmer_diacritics` — Khmer diacritization (`khmer-diacritics` repo has parallel data).
- `amharic_morphology` — Amharic morphology segmentation.
- `arabic_named_entities` — if NER maps emerge.

### 6.4 Documentation drift

- `docs/` in each repo regenerates from manifest: list of supported
  tasks, current versions, model sizes, accuracy benchmarks.
- IS-1 specification updated when grammar changes.
- `interscript.org/docs` renders live model status.

### 6.5 Observability

Once Phase 5 telemetry is stable:
- Dashboard: p50/p99 latency per model.
- Anomaly detection: DER spike on rolling window.
- SLO alerts: any model with p99 > 500 ms flagged for optimization.

## Acceptance

- [ ] Quarterly retrain cron job runs without manual intervention.
- [ ] SOTA tracker PR opened quarterly.
- [ ] New task onboarding documented end-to-end (one new task landed via this path).
- [ ] Telemetry dashboard live.

## Open questions

1. **Auto-bump patch versions**: is "CI green = auto-merge" safe enough? Risk: silent regressions that pass tests but fail real usage.
2. **Funding Modal compute**: is the user paying, or is this on a free Modal tier? Affects parallelism options.
3. **Multilingual backbone**: do we want one shared encoder + per-task decoders (NLLB pattern)? Major architecture shift. Defer to Phase 7 if data justifies.

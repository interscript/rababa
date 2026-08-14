# 38 — Per-Epoch Metrics Logging

## Problem

Currently `_status.json` only stores stage-level done/error. Per-epoch
training metrics (loss, val_loss, learning_rate) go to stdout via
`print(f"[epoch {epoch}]...")` but aren't captured in any structured form.

This caused Hebrew v0.6.0 NaN to be invisible until eval — the model
silently diverged between epoch 5 and epoch 9, but we had no way to see it.

## Fix

Add a `MetricsLogger` that writes per-epoch metrics to a structured JSONL
file on the volume:

```
/checkpoints/{task}/run-001/metrics.jsonl
```

Each line is a JSON object: `{"epoch": 0, "train_loss": 4.2, "val_loss": 4.5,
"learning_rate": 0.0003, "ts": 1234567890}`.

The supervised + pretrain + MTP loops already call `log_fn(TrainMetrics)`.
We just need to wire `log_fn` to also write to metrics.jsonl.

## Architecture

```
VolumeLogger (existing)
  ↓ log(msg: str)
  ↓ writes timestamped line to log file

MetricsLogger (NEW)
  ↓ log(metrics: TrainMetrics)
  ↓ writes JSON object to metrics.jsonl
  ↓ syncs to volume on close
```

The two loggers are siblings, both created in `run_sota_pipeline`.
The training loop's `log_fn` callable becomes a tuple of both.

## Files

- `src/rababa/training/metrics.py` (NEW) — `MetricsLogger` class.
- `src/rababa/training/resume.py` — add `MetricsLogger` next to `VolumeLogger`.
- `modal_app.py` — instantiate `MetricsLogger` in `run_sota_pipeline`.
- `tests/training/test_metrics.py` (NEW) — specs.

## Acceptance

- After pretrain completes, `/checkpoints/{task}/run-001/metrics.jsonl`
  exists with one line per epoch.
- Each line has: epoch, train_loss, val_loss, learning_rate, ts.
- Validation: load file, parse, verify N epochs.
- Specs pass.

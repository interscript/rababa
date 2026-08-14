# 14 — Performance profiling

## Why
Training cost is the bottleneck for v0.5.0 techniques. We don't know
which knob matters most: batch size, fp16 vs bf16, gradient accumulation,
sequence padding, dataloader workers.

## Tasks

### 14.1 Modal profiler (`scripts/profile_train.py`)
- Run 1 epoch with each config variant; report `examples/sec`.
- Variants: bf16/fp32, batch 16/32/64/128, workers 0/2/4/8.
- Outputs a CSV.

### 14.2 Padding optimization
- Current collate pads to batch max; sequences vary widely in length.
- New sampler: `LengthBucketSampler` groups similar-length sequences
  → less padding → ~30% throughput win.

### 14.3 ONNX inference benchmark
- `scripts/benchmark_onnx.py`: time per inference for batch 1/8/32.
- Compare fp32 vs int8 (where applicable).

## Acceptance
- [ ] Profile CSV identifies the fastest config.
- [ ] LengthBucketSampler improves examples/sec by ≥ 20%.

## Files
- `scripts/profile_train.py` (new)
- `src/rababa/training/sampler.py` (new)
- `scripts/benchmark_onnx.py` (new)

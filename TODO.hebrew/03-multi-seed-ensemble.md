# 03 — Multi-seed ensemble (Hebrew)

Same recipe as Arabic 02. Trains 3 Hebrew seeds in parallel, distills
into one shipping model.

## Tasks

### 3.1 Launch 3-seed parallel train
- `python scripts/train_seeds.py --task rababa_hebrew --n-seeds 3`
- Each seed writes to `/checkpoints/rababa_hebrew/run-{seed:03d}/best.pt`
- Modal dispatches 3 containers in parallel via `.starmap()`

### 3.2 Distill ensemble → single student
- Load all 3 teachers
- Train fresh student with `(1-α)·CE(gold) + α·KL(teacher_avg)`
- α anneal: 0.5 → 0 over training
- Output: `/checkpoints/rababa_hebrew/run-distill/best.pt`

### 3.3 Re-export ONNX + TFLite from distilled student
- Replace v0.1.0 artifacts.

## Acceptance
- [ ] 3 seeds train without OOM/conflict on Modal
- [ ] Distilled student DER < best single-seed DER by ≥ 3%

## Files
- (no new code — uses `scripts/train_seeds.py` + `training/distill.py`)
- `scripts/distill_hebrew.sh` (orchestrator wrapper, new)

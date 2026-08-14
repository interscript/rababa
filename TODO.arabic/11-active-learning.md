# 11 — Active learning

## Why
Some val examples have consistently high loss. Those patterns are
underrepresented in training. Active learning: surface those patterns,
target data collection for them (e.g. find more sentences with rare
haraqat classes in arwiki, manually verify, add to train).

## Tasks

### 11.1 Hardness miner (`src/rababa/training/active_learning.py`)
- `mine_hard_examples(model, loader, top_k=1000) -> list[(Example, loss)]`.
- Sort val examples by per-example loss, return top-K hardest.

### 11.2 Pattern analyzer
- Cluster hard examples by haraqat pattern (e.g. "all top-loss examples
  involve Sukun before Shaddah").
- Output a report listing the top patterns to mine.

### 11.3 Modal entrypoint
- `active_learning --task rababa_arabic_pro --n 1000`.
- Writes `active_learning_report.json` to `/models/{task}/`.

## Acceptance
- [ ] `mine_hard_examples` returns examples with mean loss > 2× val mean.
- [ ] Pattern report identifies ≥ 3 actionable patterns.

## Files
- `src/rababa/training/active_learning.py` (new)
- `scripts/active_learning.py` (new)
- `tests/training/test_active_learning.py` (new)

# 10 — Engram episodic memory (DeepSeek V4)

## Why
DS4's Engram (arXiv:2601.07372) is an episodic memory store that
retrieves relevant past training examples at each step. For our
use case, this addresses class imbalance: rare haraqat combos
(Shaddah+Kasratan — <0.1% of training) get reinforced by retrieving
similar contexts from the episodic store.

## Architecture

```
forward(x):
    1. Standard encoder forward.
    2. Query engram with hidden states → retrieve top-K similar past examples.
    3. Concatenate retrieved hidden states to current hidden.
    4. Small projection layer → fed to head.
```

Engram store is updated during training (FIFO + importance sampling).

## Tasks

### 10.1 Engram module (`src/rababa/models/engram.py`)
- `Engram(dim, capacity=10000, top_k=4)`.
- Stores `(hidden, label)` pairs; retrieval by cosine similarity.
- Differentiable end-to-end.

### 10.2 ModernCharTransformerEngram
- New arch variant: `arch: "modern_engram"`.
- Same encoder, plus engram read/write per layer.

### 10.3 Wire into training loop
- Engram is populated by current batch + sampled from past batches.
- Adds one extra forward pass per step (small).

## Acceptance
- [ ] Engram at capacity=10K fits in A100 memory alongside the model.
- [ ] Per-class DER on rare haraqat improves by ≥ 5%.
- [ ] Common-class DER unchanged (no regression).

## Files
- `src/rababa/models/engram.py` (new)
- `src/rababa/models/modern.py` (add engram variant)
- `tests/models/test_engram.py` (new)

## Open questions
- Capacity: 10K is from the paper. For our smaller model, sweep 1K/5K/10K.

## CORRECTION (2026-09-11, from the DeepSeek-V4.1-Flash report §2.4.2)

The mechanism described above (retrieving similar past training
examples) is NOT Engram. The actual design (arXiv:2601.07372):
hash-addressed n-gram embedding tables summed into the forward pass —
no example retrieval, no similarity search, no FIFO store.

- addresses = n-gram hashes of the input token sequence
- orders {2,3,4}, 8 hash heads, 2048-dim per order
- table sizes are distinct primes (~16M entries per head in DS's
  196B-parameter deployment)
- FP8 storage, modules placed at layers 1 and 14
- 5x learning rate for the tables; table updates use the
  Sinkhorn-balanced momentum rule (momentum + alternating row/col
  normalization), not Adam
- effect: memorization decoupled from compute — deterministic lookup,
  prefetchable, no bandwidth interaction with attention

The class-imbalance rationale above (rare haraqat reinforced by
retrieval) does not transfer to the real mechanism; the applicable
rationale is lexical idiom: haraqat are strongly n-gram-driven, and a
lookup table captures idiosyncratic lexical patterns without
compute. Any wire-in must be rebuilt against the real design; the
browser size ceiling (int8 4M x 64-dim ~ 64 MB) bounds the table.
Tracked at TODO.impl/10.

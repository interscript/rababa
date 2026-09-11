# 09 — External-reference doc corrections (P2)

Two rababa notes describe mechanisms incorrectly relative to their
sources. Both were written from summaries, not the papers; the
DeepSeek V4.1-Flash report quotes the actual designs.

## Fixes

1. **Engram (TODO.arabic/10, TODO.hebrew/11)**: described as
   "retrieving similar past training examples" (episodic replay).
   The actual mechanism (arXiv:2601.07372; DS V4.1-Flash §2.4.2) is
   hash-addressed n-gram embedding tables summed into the forward
   pass: orders {2,3,4}, 8 hash heads, 2048-dim per order, distinct
   prime table sizes, FP8 storage, modules at layers 1 and 14, 5x LR,
   tables updated by the Sinkhorn-balanced rule. Memorization
   decoupled from compute — no example retrieval at all.
2. **mHC single-pass shift (TODO.arabic/19)**: add the report's
   deployable variant — each block consumes the PREVIOUS block's
   input-mixing coefficients (A_{l-1}), killing the kernel dependency
   and halving activation memory traffic at "negligible degradation"
   (§2.4.1). Adopt if n-stream mHC resumes.

## Acceptance

- [ ] Both Engram notes corrected with the real mechanism + config
- [ ] mHC note carries the single-pass shift
- [ ] Both land in the same rababa PR as the TODO.impl ledger

## Status

- [ ] Edits

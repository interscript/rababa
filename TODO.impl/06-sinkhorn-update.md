# 06 — Sinkhorn-balanced momentum update for embedding/head (P2)

Source: TODO.qwen-next/10 §6, DeepSeek Alg. 1. Replaces Adam for
embedding tables and the prediction head: Nesterov momentum,
alternating row/column L2 normalization (odd K steps), near-zero row
masking, sqrt(n) RMS conversion, learning-rate correction gamma=0.18,
no weight decay. Momentum-only state (vs Adam's two moments) and
"empirically outperforming Adam" at their scale — unproven at ours,
which is the point of measuring it.

We already own log-domain Sinkhorn machinery from the mHC work
(direct division NaNs for ~10% of inits — the log-domain lesson
carries to the row/col normalizations here if magnitudes get small).

## Deliverables

1. `src/gpu/sinkhorn_update.py`: the update as a pure function over
   (weight, grad, momentum state) + an optimizer wrapper selecting it
   for embedding/head parameter patterns by name — OCP: the training
   loop keeps AdamW/Muon dispatch, this registers as another
   treatment.
2. Unit tests with real tensors: masked rows stay masked; unit
   row-RMS after balancing; momentum state evolution; the sqrt(n)
   and gamma factors against a hand-computed 3x2 case.
3. Config-gated use in a distill spec (one run) if the owner wants
   the quality datapoint; otherwise the implementation + tests land
   ready.

## Acceptance

- [ ] Update implemented exactly per Alg. 1 (K=11, tau=1e-3,
      eps=1e-20, gamma=0.18, Nesterov beta, no weight decay)
- [ ] Unit tests green on real tensors
- [ ] Spec entry (off by default) for the optional measurement run
- [ ] No change to any shipped run's reproducibility

## Status

- [ ] Implementation
- [ ] Tests
- [ ] Optional run decision

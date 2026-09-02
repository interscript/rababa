# 40 — MoE Forward Efficiency (Top-K Only Compute)

## Problem

Current `LatentMoE.forward` computes ALL experts on ALL tokens, then gathers
top-K outputs:

```python
all_expert_outs = torch.stack([expert(flat) for expert in self.experts], dim=1)
topk_outs = all_expert_outs.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
```

For n_experts=16, top_k=2, this is **8x wasted compute**. On Hebrew/Arabic
where MoE is the FFN, this is the largest single perf bottleneck.

## Fix

Replace with gather-scatter that computes only used experts:

```python
# For each (token, top-k slot), find the expert it routes to and gather
# that expert's weights. Use grouped MM via einsum.
out = torch.zeros_like(flat)
for k in range(self.top_k):
    expert_ids = topk_idx[:, k]  # (BT,)
    # For each expert e, find tokens routed to it at slot k.
    for e in range(self.n_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        x_subset = flat[mask]
        out_subset = self.experts[e](x_subset)
        out[mask] += out_subset * topk_probs[mask, k].unsqueeze(-1)
```

For our scale (n_experts=16, BT≤4096), the loop is fast enough. For larger
scales, use `grouped_gemm` from megablocks/research kernels.

Expected speedup: 4-8x on MoE FFN, ~30% on full training step.

## Files

- `src/rababa/models/moe.py:LatentMoE.forward` — replace gather with scatter.
- `tests/models/test_moe.py` — add perf spec: top_k=2/n=4 should be ~2x faster
  than the all-experts path on a fixed input.

## Acceptance

- Output identical to current implementation (within fp tolerance).
- Forward time reduced ~4x for typical n_experts=16, top_k=2.
- All existing specs pass.

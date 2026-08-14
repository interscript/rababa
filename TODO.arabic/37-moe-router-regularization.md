# 37 — MoE Router Regularization + Optimizer Routing Fix

## Problem

Hebrew v0.6.0 (first attempt) had MoE router weights explode to norm=131
(init ≈19). Root cause: the Muon optimizer's Newton-Schulz orthogonalization
over-amplifies router weights. Routers output softmax → tiny logit changes
have large effect on routing decisions → unstable training.

Additionally, the current `LatentMoE.forward` computes ALL experts then
gathers top-K. For n_experts=16, top_k=2, this wastes 8x compute.

## Fix

### Part A: Route router weights to AdamW (not Muon)

In `MuonAdamWHybrid.__init__`, exclude `moe.router` from Muon param group:

```python
elif p.ndim == 2 and "embedding" not in name and "norm" not in name \
     and "moe.router" not in name and "router" not in name:
    muon_params.append(p)
else:
    adam_params.append(p)
```

Routers are sensitive to orthogonalization; they belong with AdamW along
with embeddings, norms, biases, and 1D params.

### Part B: Add max-norm constraint to router (defense-in-depth)

Add `router_max_norm` parameter to `LatentMoE`. After each forward, clamp
router weight norm if it exceeds the threshold:

```python
def _clamp_router_norm(self) -> None:
    if self.router_max_norm is None:
        return
    with torch.no_grad():
        norm = self.router.weight.norm()
        if norm > self.router_max_norm:
            scale = self.router_max_norm / norm.clamp_min(1e-8)
            self.router.weight.mul_(scale)
```

Called from `forward()` after storing routing probs.

### Part C: Top-K only expert computation (performance)

Replace the current "compute all experts, gather top-K" with proper
gather-scatter:

```python
# For each token, gather only its top_k experts' weights + compute
flat = x.reshape(B * T, D)
router_logits = self.router(flat)
probs = F.softmax(router_logits, dim=-1)
topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
topk_probs /= topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

# Compute only the experts that are actually used.
# Use einsum-based scatter to avoid O(N_experts) compute.
out = torch.zeros_like(flat)
for k in range(self.top_k):
    expert_outputs = torch.stack([
        self.experts[idx](flat[i]) for i, idx in enumerate(topk_idx[:, k])
    ], dim=0)  # This is O(BT) calls — still bad
    ...
```

The truly efficient version uses grouped MM (torch.scatter + batched mm).
For our scale (n_experts=8-32, BT≤4096), the simpler version is:

```python
# Compute every expert once on the full batch, then gather.
# Same compute as current, just cleaner code. Real perf win needs
# megablocks/grouped_gemm ops.
```

Defer the truly efficient implementation until we hit perf bottleneck.
For now, the simple "compute all + gather" is acceptable.

## Files

- `src/rababa/training/optim.py:MuonAdamWHybrid.__init__` — exclude router.
- `src/rababa/models/moe.py:LatentMoE.__init__` — add `router_max_norm` param.
- `src/rababa/models/moe.py:LatentMoE.forward` — call `_clamp_router_norm()`.
- `tests/models/test_moe.py` — add spec: router norm bounded.
- `tests/training/test_optim.py` (new) — add spec: router in AdamW group.

## Acceptance

- Router weights stay bounded (norm < 50 after full training).
- All MoE specs pass.
- Hebrew v0.6.0 doesn't NaN.

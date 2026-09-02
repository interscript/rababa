"""Modern char-level Transformer encoder — K3/DS4 stack.

Architecture choices drawn straight from current frontier papers:
  - **mHC** (Manifold-Constrained Hyper-Connections) — DeepSeek V4, arXiv:2512.24880
  - **AttnRes** (Attention Residuals) — Kimi K3, arXiv:2607.24653
  - **RoPE** (Su et al., 2021) — used by DS4 and K3
  - **SDPA** — PyTorch 2.x scaled_dot_product_attention (Flash / mem-efficient)
  - **RMSNorm** — DS4 + K3 default (drops LayerNorm's mean-centering)
  - **SwiGLU FFN** — Llama / DS / Kimi default

The optimizer side of the K3/DS4 stack (Per-Head Muon + QK-Clip) lives
in `training/optim.py` and is wired in via the pretrain / supervised
loops. See task #182.

Differences from the baseline CharTransformer (student.py):
  - No learned positional embedding — RoPE rotates Q/K in attention.
  - mHC replaces standard `x + sublayer(x)` residual. Multi-stream mix
    with Sinkhorn-Knopp-normalized mixing matrix gives an identity
    guarantee that prevents residual collapse during from-scratch
    pretraining.
  - AttnRes passes each layer's attention output to the next layer's
    attention input, improving information flow across depth.
  - SwiGLU FFN instead of GELU MLP (faster convergence, modern default).
  - RMSNorm instead of LayerNorm (no mean-centering, cheaper, standard
    in DS4 / K3).
  - SDPA used directly so Flash / memory-efficient kernels auto-trigger.

Same Diacritizer protocol as CharTransformer — drop-in replacement
selected by `cfg.model.arch: "modern"`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE


# ---- Rotary positional embedding --------------------------------------


class RotaryEmbedding(nn.Module):
    """Pre-computes cos/sin tables for RoPE. Buffers move with the module."""

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to Q and K. Both: (B, H, T, D_head)."""
    q_r = q * cos + _rotate_half(q) * sin
    k_r = k * cos + _rotate_half(k) * sin
    return q_r, k_r


# ---- RMSNorm (DS4 + K3 default) ---------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm — no mean-centering, no bias.

    Standard in DeepSeek V4 and Kimi K3. Slightly cheaper than LayerNorm
    and empirically equivalent or better for transformer training.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---- Sinkhorn-Knopp projection for mHC --------------------------------


# ---- Sinkhorn-Knopp projection for mHC --------------------------------


def sinkhorn_knopp(mat: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project mat onto the Birkhoff polytope (doubly-stochastic).

    Numerically safe log-domain Sinkhorn. Avoids division-by-near-zero
    failures that the direct formulation produces for matrices with
    small column sums (can occur with our `eye(2) + 0.01 * randn` init).
    """
    log_m = torch.log(mat.abs().clamp_min(1e-8))
    for _ in range(iters):
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
    return torch.exp(log_m)


# ---- Manifold-Constrained Hyper-Connections --------------------------


class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connections (DeepSeek V4, simplified 2-stream).

    Standard residual: x + sublayer_out.
    mHC residual:      M @ [x, sublayer_out]   where M is SK-normalized 2x2.

    The SK projection forces M onto the Birkhoff polytope, giving an
    identity guarantee that prevents residual-stream collapse. The raw
    matrix is learned; gradients flow through SK iterations.

    See `MHCN` for the N-stream generalization (DS4 uses up to 4 streams).
    """

    def __init__(self, sk_iters: int = 20) -> None:
        super().__init__()
        self.sk_iters = sk_iters
        # Start near identity so initial behavior matches standard residual.
        raw = torch.eye(2) + 0.01 * torch.randn(2, 2)
        self.mix_raw = nn.Parameter(raw)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        streams = torch.stack((x, sublayer_out), dim=2)  # (B, T, 2, D)
        m = sinkhorn_knopp(self.mix_raw, self.sk_iters)
        mixed = torch.einsum("ij,btid->btjd", m, streams)
        return mixed[:, :, 0, :]


class MHCN(nn.Module):
    """N-stream Manifold-Constrained Hyper-Connections (DS4-style).

    Generalizes `MHC` from 2 streams (residual + 1 sublayer) to N streams
    (residual + N-1 sublayers). The mixing matrix is N×N, SK-normalized
    to be doubly-stochastic.

    For encoder layer (attn + ffn): 3 streams.
    For decoder layer (self + cross + ffn): 4 streams.

    The "carry forward" stream (index 0) is the residual path. Other
    streams contribute to the mix but aren't propagated as the residual
    for the next layer.
    """

    def __init__(self, n_streams: int, sk_iters: int = 20) -> None:
        super().__init__()
        assert n_streams >= 2, f"n_streams must be ≥ 2, got {n_streams}"
        self.n_streams = n_streams
        self.sk_iters = sk_iters
        raw = torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams)
        self.mix_raw = nn.Parameter(raw)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        """Mix N streams via SK-normalized matrix. Returns the first stream after mixing."""
        assert len(streams) == self.n_streams, (
            f"MHCN expected {self.n_streams} streams, got {len(streams)}"
        )
        stacked = torch.stack(streams, dim=2)  # (B, T, N, D)
        m = sinkhorn_knopp(self.mix_raw, self.sk_iters)
        mixed = torch.einsum("ij,btid->btjd", m, stacked)
        return mixed[:, :, 0, :]  # carry forward the first stream


# ---- Modern encoder layer ---------------------------------------------


class ModernEncoderLayer(nn.Module):
    """Pre-norm encoder layer with SwiGLU/MoE FFN, mHC residuals, exposes attn_out for AttnRes.

    SOTA techniques supported (all optional, off by default for backward compat):
      - `ffn_type="moe"`: LatentMoE FFN instead of SwiGLU.
      - `kv_heads < heads`: Grouped Query Attention (Qwen3, Llama-3).
      - `qk_norm=True`: RMSNorm on Q/K before attention dot product (Qwen-Max).
      - `kda=True`: per-layer attention bias (Kimi K3 KDA).
      - `norm_type="zero_centered"`: Zero-centered RMSNorm (Qwen3.5). gamma init=0,
        `out = x * rsqrt(mean(x²)+eps) * gamma + x` — identity at init, better
        gradient signal for deep stacks. Default "rmsnorm" (standard, gamma init=1).
      - `swiglu_clamp_max=N` (DS-V4-Flash §4.2.3): clamp SwiGLU linear path to
        [-N, N], cap gate at N. Eliminates outliers → stable training.
        Default None (off for backward compat). V4-Flash uses 10.0.
      - `use_sink=True` (DS-V4-Flash §2.3.3, Eq. 27): learnable per-head sink logit
        added to softmax denominator. Prevents first-token overattention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        sk_iters: int = 20,
        ffn_type: str = "swiglu",
        moe_config: dict | None = None,
        kv_heads: int | None = None,
        qk_norm: bool = False,
        kda: bool = False,
        norm_type: str = "rmsnorm",
        swiglu_clamp_max: float | None = None,
        use_sink: bool = False,
        resformer_lambda1: float | None = None,
        resformer_lambda2: float | None = None,
    ) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        assert heads % self.kv_heads == 0, (
            f"heads ({heads}) must be divisible by kv_heads ({self.kv_heads})"
        )
        self.head_dim = dim // heads
        self.dim = dim
        self.ffn_type = ffn_type
        self.norm_type = norm_type
        self.swiglu_clamp_max = swiglu_clamp_max
        self.use_sink = use_sink

        # ResFormer value residual (arXiv:2410.17897, ACL 2025).
        # V_n = λ_1 · V_1 + λ_2 · (H_{n-1} · W_V_n)
        # V_1 is supplied by the transformer's forward pass (cached from layer 0).
        # λ_1 / λ_2 are learnable per-layer scalars; when only one is set, the
        # other defaults to 0.5 (Identity-ResFormer init in the paper).
        # Sparse-ResFormer variant: set resformer_lambda1=None on layers that
        # should NOT receive V_1 (only later layers benefit per paper Fig. 5).
        self.use_resformer = resformer_lambda1 is not None
        if self.use_resformer:
            lam1 = float(resformer_lambda1)  # type: ignore[arg-type]
            lam2 = float(resformer_lambda2) if resformer_lambda2 is not None else 0.5
            self.resformer_lambda1 = nn.Parameter(torch.tensor(lam1))
            self.resformer_lambda2 = nn.Parameter(torch.tensor(lam2))

        self.norm1 = self._build_norm(dim, norm_type)
        if self.kv_heads == self.heads:
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        else:
            # GQA: separate Q (full) + K, V (reduced) projections.
            self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
            self.kv_proj = nn.Linear(dim, 2 * self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # QK-Norm: RMSNorm on per-head Q and K vectors (Qwen-Max).
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # KDA: per-layer attention bias (K3).
        self.use_kda = kda
        if kda:
            from .kda import KDABias
            self.kda_bias = KDABias(init_value=0.0)

        # Attention Sink (DS-V4-Flash §2.3.3, Eq. 27): per-head learnable logit
        # added to softmax denominator as exp(sink_logit), allowing attention mass
        # to "leak" to a virtual sink token. Init=0 → exp(0)=1 contribution.
        if use_sink:
            self.sink_logit = nn.Parameter(torch.zeros(heads))

        self.norm2 = self._build_norm(dim, norm_type)
        if ffn_type == "moe":
            from .moe import LatentMoE
            mc = moe_config or {}
            self.moe = LatentMoE(
                dim=dim,
                n_experts=mc.get("n_experts", 32),
                expert_dim=mc.get("expert_dim", ff_dim),
                top_k=mc.get("top_k", 4),
                shared_experts=mc.get("shared_experts", 0),
                swiglu_clamp_max=swiglu_clamp_max,
                affinity_type=mc.get("affinity_type", "softmax"),
            )
        else:
            self.w_gate = nn.Linear(dim, ff_dim, bias=False)
            self.w_up = nn.Linear(dim, ff_dim, bias=False)
            self.w_down = nn.Linear(ff_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.mhc_attn = MHC(sk_iters=sk_iters)
        self.mhc_ff = MHC(sk_iters=sk_iters)
        # Init attention linears for stability (small normal init).
        attn_linears: list[nn.Linear] = []
        if self.kv_heads == self.heads:
            attn_linears = [self.qkv]
        else:
            attn_linears = [self.q_proj, self.kv_proj]
        attn_linears.append(self.out_proj)
        for ln in attn_linears:
            nn.init.normal_(ln.weight, mean=0.0, std=0.02)

    def _build_norm(self, dim: int, norm_type: str) -> nn.Module:
        if norm_type == "zero_centered":
            from .zero_centered_rmsnorm import ZeroCenteredRMSNorm
            return ZeroCenteredRMSNorm(dim)
        if norm_type != "rmsnorm":
            raise ValueError(f"unknown norm_type: {norm_type!r} (expected 'rmsnorm' or 'zero_centered')")
        return RMSNorm(dim)

    def _attention(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   key_padding_mask: torch.Tensor | None,
                   v1: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        if self.kv_heads == self.heads:
            qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            # GQA path.
            q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
            kv = self.kv_proj(x).reshape(B, T, 2, self.kv_heads, self.head_dim)
            k, v = kv.unbind(dim=2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            group_size = self.heads // self.kv_heads
            if group_size > 1:
                k = k.repeat_interleave(group_size, dim=1)
                v = v.repeat_interleave(group_size, dim=1)
        # Cache this layer's pre-residual V for downstream ResFormer layers.
        # The first layer's cached V becomes V_1; subsequent layers add it
        # via the value residual when use_resformer=True and v1 is supplied.
        v_pre_residual = v
        if self.use_resformer and v1 is not None:
            # ResFormer (arXiv:2410.17897): V_n = λ_1·V_1 + λ_2·V_n
            v = self.resformer_lambda1 * v1 + self.resformer_lambda2 * v
        # Expose V_1 to the transformer's forward loop via attribute.
        # Set only on the first layer (when v1 is None) so downstream layers
        # pick it up from the cached slot.
        if v1 is None:
            self._v_first = v_pre_residual
        q, k = apply_rope(q, k, cos, sin)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        if self.use_sink:
            attn = self._attention_with_sink(q, k, v, attn_mask)
        elif self.use_kda:
            from .kda import softmax_with_kda
            attn = softmax_with_kda(
                q, k, v,
                kda_bias=self.kda_bias(),
                attn_mask=attn_mask,
                dropout_p=0.0 if not self.training else self.dropout.p,
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=0.0 if not self.training else self.dropout.p,
            )
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        return self.out_proj(attn)

    def _attention_with_sink(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """SDPA-equivalent attention with DS-V4-Flash attention sink (Eq. 27).

        Implements s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(z'_h))
        by appending a virtual KV position with logit z'_h and value 0.
        The virtual position contributes exp(z'_h) to the denominator but 0 to
        the numerator (since its value is 0), giving exactly the sink formula.
        """
        H = self.heads
        T = q.size(-2)
        # scores: (B, H, T_q, T_k)
        scale = q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if key_attn_mask is not None:
            # key_attn_mask: (B, 1, 1, T_k) bool, True = mask out.
            scores = scores.masked_fill(key_attn_mask, float("-inf"))
        # Append virtual sink column with logit = sink_logit[h] per head.
        # sink_logit: (H,) → (1, H, T_q, 1) broadcasting
        sink_col = self.sink_logit.view(1, H, 1, 1).expand(scores.size(0), H, T, 1)
        extended = torch.cat([scores, sink_col], dim=-1)  # (B, H, T_q, T_k+1)
        # Sink column is never masked (it's always reachable).
        attn = torch.softmax(extended, dim=-1)
        # Drop the sink column from the output weights.
        attn_weights = attn[..., :T]  # (B, H, T_q, T_k)
        if self.training and self.dropout.p > 0:
            attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "moe":
            return self.moe(x)
        from .swiglu import swiglu
        return self.w_down(swiglu(self.w_gate(x), self.w_up(x), clamp_max=self.swiglu_clamp_max))

    def moe_load_balance_loss(self) -> torch.Tensor:
        """Auxiliary load-balance loss if FFN is MoE; zero otherwise."""
        if self.ffn_type == "moe":
            return self.moe.load_balance_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prev_attn: torch.Tensor | None = None,
        v1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out = self._attention(self.norm1(x), cos, sin, key_padding_mask, v1=v1)
        # AttnRes: add previous layer's attention output before mHC mixing.
        if prev_attn is not None:
            attn_out = attn_out + prev_attn
        x = self.mhc_attn(x, attn_out)
        ff_out = self._ffn(self.norm2(x))
        x = self.mhc_ff(x, ff_out)
        return x, attn_out


# ---- Modern char transformer ------------------------------------------


class ModernCharTransformer(nn.Module):
    """Char-level Transformer encoder + linear head.

    Replaces CharTransformer (student.py) when cfg.model.arch == "modern".
    Same Diacritizer protocol: forward_heads() returns single-element list.
    """

    def __init__(
        self,
        input_vocab_size: int = INPUT_VOCAB_SIZE,
        target_vocab_size: int = TARGET_VOCAB_SIZE,
        dim: int = 768,
        layers: int = 12,
        heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        with_seg_head: bool = False,
        kv_heads: int | None = None,
        qk_norm: bool = False,
        kda: bool = False,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        moe_config: dict | None = None,
        swiglu_clamp_max: float | None = None,
        use_sink: bool = False,
        resformer: dict | None = None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len
        self.head_dim = dim // heads
        self.with_seg_head = with_seg_head

        self.embedding = nn.Embedding(input_vocab_size, dim, padding_idx=pad_id)
        self.rotary = RotaryEmbedding(self.head_dim, max_len=max_len, base=rope_base)
        # ResFormer config (arXiv:2410.17897). Two modes:
        #   - "all": every layer ≥1 receives V_1 with init λ_1 = λ_2 = 0.5
        #     (Learnable-ResFormer init).
        #   - "sparse": only the last K layers receive V_1 with init λ_1=5,
        #     λ_2=1 (Sparse-ResFormer recipe from Table 3).
        resformer_mode = (resformer or {}).get("mode", "off")
        resformer_n = int((resformer or {}).get("n_last_layers", max(1, layers // 3)))
        resformer_lambda1 = (resformer or {}).get("lambda1", 0.5)
        resformer_lambda2 = (resformer or {}).get("lambda2", 0.5)
        self.layers = nn.ModuleList()
        for i in range(layers):
            if resformer_mode == "all":
                lam1 = resformer_lambda1 if i >= 1 else None
                lam2 = resformer_lambda2
            elif resformer_mode == "sparse":
                # Only last n_last_layers get V_1; paper recommends λ_1=5.
                is_sparse_layer = i >= (layers - resformer_n) and i >= 1
                lam1 = resformer_lambda1 if is_sparse_layer else None
                lam2 = resformer_lambda2
            else:
                lam1 = None
                lam2 = None
            self.layers.append(
                ModernEncoderLayer(
                    dim, heads, ff_dim,
                    dropout=dropout, sk_iters=sk_iters,
                    ffn_type=ffn_type, moe_config=moe_config,
                    kv_heads=kv_heads, qk_norm=qk_norm, kda=kda,
                    norm_type=norm_type,
                    swiglu_clamp_max=swiglu_clamp_max,
                    use_sink=use_sink,
                    resformer_lambda1=lam1,
                    resformer_lambda2=lam2,
                )
            )
        self.final_norm = self._build_final_norm(dim, norm_type)
        self.head = nn.Linear(dim, target_vocab_size)
        # Multi-task aux head (T1.2): word-segmentation boundary prediction.
        # Labels are trivially derived from input (1 at word boundaries, 0 elsewhere)
        # so no external labeler is needed — the value is encoder regularization.
        # POS head deferred to v2 (needs external POS tagger for labels).
        if with_seg_head:
            self.seg_head = nn.Linear(dim, 2)

    def _build_final_norm(self, dim: int, norm_type: str) -> nn.Module:
        if norm_type == "zero_centered":
            from .zero_centered_rmsnorm import ZeroCenteredRMSNorm
            return ZeroCenteredRMSNorm(dim)
        return RMSNorm(dim)

    def forward_encoder(self, src: torch.Tensor) -> torch.Tensor:
        """Embed tokens, run encoder, return final hidden states.

        Shared with the MLM pretraining head — same protocol as
        CharTransformer.forward_encoder so existing pretrain.py works
        unchanged.
        """
        batch_size, seq_len = src.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        key_padding_mask = src == self.pad_id
        x = self.embedding(src)
        cos, sin = self.rotary(seq_len)
        prev_attn: torch.Tensor | None = None
        v1: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            # After layer 0 runs, pick up its cached V for ResFormer downstream.
            x, prev_attn = layer(x, cos, sin, key_padding_mask, prev_attn, v1=v1)
            if i == 0 and hasattr(layer, "_v_first"):
                v1 = layer._v_first
        return self.final_norm(x)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_encoder(src))

    def forward_heads(self, src: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        # Multi-task: [haraqat, seg] when seg head enabled; else [haraqat].
        h = self.forward_encoder(src)
        out = [self.head(h)]
        if self.with_seg_head:
            out.append(self.seg_head(h))
        return out

    def head_names(self) -> list[str]:
        return ["output", "seg"] if self.with_seg_head else ["output"]


def build_modern_student(cfg: dict[str, Any]) -> ModernCharTransformer:
    """Factory: build ModernCharTransformer from a config dict."""
    m = cfg.get("model", {})
    return ModernCharTransformer(
        input_vocab_size=m.get("input_vocab_size", INPUT_VOCAB_SIZE),
        target_vocab_size=m.get("target_vocab_size", TARGET_VOCAB_SIZE),
        dim=m.get("dim", 768),
        layers=m.get("layers", 12),
        heads=m.get("heads", 12),
        ff_dim=m.get("ff_dim", 3072),
        dropout=m.get("dropout", 0.1),
        max_len=m.get("max_len", 512),
        rope_base=m.get("rope_base", 10000.0),
        sk_iters=m.get("sk_iters", 20),
        with_seg_head=m.get("with_seg_head", False),
        kv_heads=m.get("kv_heads", None),
        qk_norm=m.get("qk_norm", False),
        kda=m.get("kda", False),
        norm_type=m.get("norm_type", "rmsnorm"),
        ffn_type=m.get("ffn_type", "swiglu"),
        moe_config=m.get("moe", None),
        swiglu_clamp_max=m.get("swiglu_clamp_max", None),
        use_sink=m.get("use_sink", False),
        resformer=m.get("resformer", None),
    )


# ---- Modern multi-head (Hebrew) --------------------------------------


class ModernMultiHeadCharTransformer(nn.Module):
    """Modern encoder + multiple linear heads (Hebrew: niqqud, dagesh, sin).

    Same encoder body as `ModernCharTransformer` (RoPE + SDPA + mHC +
    AttnRes + RMSNorm + SwiGLU). Only the head differs: a `ModuleList`
    of linear projections, one per output category.

    Encoder weights are key-compatible with `ModernCharTransformer` so a
    single MLM-pretrained encoder can fine-tune into either the
    single-head Arabic student or this multi-head Hebrew student.
    """

    def __init__(
        self,
        input_vocab_size: int,
        head_sizes: list[int],
        dim: int = 384,
        layers: int = 6,
        heads: int = 6,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        kv_heads: int | None = None,
        qk_norm: bool = False,
        kda: bool = False,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        moe_config: dict | None = None,
        swiglu_clamp_max: float | None = None,
        use_sink: bool = False,
        resformer: dict | None = None,
    ) -> None:
        super().__init__()
        from .multi_head import OUTPUT_ORDER
        if len(head_sizes) != len(OUTPUT_ORDER):
            raise ValueError(
                f"head_sizes must have {len(OUTPUT_ORDER)} entries "
                f"({', '.join(OUTPUT_ORDER)}); got {len(head_sizes)}"
            )
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len
        self.head_dim = dim // heads
        self.head_sizes = head_sizes

        self.embedding = nn.Embedding(input_vocab_size, dim, padding_idx=pad_id)
        self.rotary = RotaryEmbedding(self.head_dim, max_len=max_len, base=rope_base)
        resformer_mode = (resformer or {}).get("mode", "off")
        resformer_n = int((resformer or {}).get("n_last_layers", max(1, layers // 3)))
        resformer_lambda1 = (resformer or {}).get("lambda1", 0.5)
        resformer_lambda2 = (resformer or {}).get("lambda2", 0.5)
        self.layers = nn.ModuleList()
        for i in range(layers):
            if resformer_mode == "all":
                lam1 = resformer_lambda1 if i >= 1 else None
                lam2 = resformer_lambda2
            elif resformer_mode == "sparse":
                is_sparse_layer = i >= (layers - resformer_n) and i >= 1
                lam1 = resformer_lambda1 if is_sparse_layer else None
                lam2 = resformer_lambda2
            else:
                lam1 = None
                lam2 = None
            self.layers.append(
                ModernEncoderLayer(
                    dim, heads, ff_dim,
                    dropout=dropout, sk_iters=sk_iters,
                    ffn_type=ffn_type, moe_config=moe_config,
                    kv_heads=kv_heads, qk_norm=qk_norm, kda=kda,
                    norm_type=norm_type,
                    swiglu_clamp_max=swiglu_clamp_max,
                    use_sink=use_sink,
                    resformer_lambda1=lam1,
                    resformer_lambda2=lam2,
                )
            )
        self.final_norm = self._build_final_norm(dim, norm_type)
        self.heads = nn.ModuleList([nn.Linear(dim, n) for n in head_sizes])

    def _build_final_norm(self, dim: int, norm_type: str) -> nn.Module:
        if norm_type == "zero_centered":
            from .zero_centered_rmsnorm import ZeroCenteredRMSNorm
            return ZeroCenteredRMSNorm(dim)
        return RMSNorm(dim)

    def forward_encoder(self, src: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = src.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        key_padding_mask = src == self.pad_id
        x = self.embedding(src)
        cos, sin = self.rotary(seq_len)
        prev_attn: torch.Tensor | None = None
        v1: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            x, prev_attn = layer(x, cos, sin, key_padding_mask, prev_attn, v1=v1)
            if i == 0 and hasattr(layer, "_v_first"):
                v1 = layer._v_first
        return self.final_norm(x)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        """Return [head_0_logits, head_1_logits, ...] in canonical order."""
        hidden = self.forward_encoder(src)
        return [head(hidden) for head in self.heads]

    def forward_heads(self, src: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        return self.forward(src, lengths)

    def head_names(self) -> list[str]:
        from .multi_head import OUTPUT_ORDER
        return list(OUTPUT_ORDER)


def build_modern_multi_head_student(cfg: dict[str, Any]) -> ModernMultiHeadCharTransformer:
    """Factory: build ModernMultiHeadCharTransformer from a config dict."""
    from ..constants_hebrew import (
        DAGESH_VOCAB_SIZE,
        INPUT_VOCAB_SIZE as HEBREW_INPUT_VOCAB_SIZE,
        NIQQUD_VOCAB_SIZE,
        SIN_VOCAB_SIZE,
    )
    m = cfg.get("model", {})
    return ModernMultiHeadCharTransformer(
        input_vocab_size=m.get("input_vocab_size", HEBREW_INPUT_VOCAB_SIZE),
        head_sizes=m.get("head_sizes", [NIQQUD_VOCAB_SIZE, DAGESH_VOCAB_SIZE, SIN_VOCAB_SIZE]),
        dim=m.get("dim", 384),
        layers=m.get("layers", 6),
        heads=m.get("heads", 6),
        ff_dim=m.get("ff_dim", 1536),
        dropout=m.get("dropout", 0.1),
        max_len=m.get("max_len", 512),
        rope_base=m.get("rope_base", 10000.0),
        sk_iters=m.get("sk_iters", 20),
        kv_heads=m.get("kv_heads", None),
        qk_norm=m.get("qk_norm", False),
        kda=m.get("kda", False),
        norm_type=m.get("norm_type", "rmsnorm"),
        ffn_type=m.get("ffn_type", "swiglu"),
        moe_config=m.get("moe", None),
        swiglu_clamp_max=m.get("swiglu_clamp_max", None),
        use_sink=m.get("use_sink", False),
        resformer=m.get("resformer", None),
    )

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


def sinkhorn_knopp(mat: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project mat onto the Birkhoff polytope (doubly-stochastic matrices).

    Alternating row / column normalization. Differentiable — gradients
    flow back through the iterations to the raw parameter.
    """
    m = mat
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-8)
    return m


# ---- Manifold-Constrained Hyper-Connections --------------------------


class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connections (DeepSeek V4, simplified 2-stream).

    Standard residual: x + sublayer_out.
    mHC residual:      M @ [x, sublayer_out]   where M is SK-normalized 2x2.

    The SK projection forces M onto the Birkhoff polytope, giving an
    identity guarantee that prevents residual-stream collapse. The raw
    matrix is learned; gradients flow through SK iterations.
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
        return mixed[:, :, 0, :]  # carry forward the first stream


# ---- Modern encoder layer ---------------------------------------------


class ModernEncoderLayer(nn.Module):
    """Pre-norm encoder layer with SwiGLU FFN, mHC residuals, exposes attn_out for AttnRes."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        sk_iters: int = 20,
    ) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim

        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = RMSNorm(dim)
        # SwiGLU: gate * up, then project down. ff_dim is the inner size.
        self.w_gate = nn.Linear(dim, ff_dim, bias=False)
        self.w_up = nn.Linear(dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.mhc_attn = MHC(sk_iters=sk_iters)
        self.mhc_ff = MHC(sk_iters=sk_iters)

    def _attention(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D_head)
        q = q.transpose(1, 2)  # (B, H, T, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # PyTorch 2.x SDPA — auto-Flash / memory-efficient kernels.
        # key_padding_mask needs shaping to (B, 1, 1, T) for SDPA broadcast.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else self.dropout.p
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        return self.out_proj(attn)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prev_attn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out = self._attention(self.norm1(x), cos, sin, key_padding_mask)
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
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len
        self.head_dim = dim // heads
        self.with_seg_head = with_seg_head

        self.embedding = nn.Embedding(input_vocab_size, dim, padding_idx=pad_id)
        self.rotary = RotaryEmbedding(self.head_dim, max_len=max_len, base=rope_base)
        self.layers = nn.ModuleList([
            ModernEncoderLayer(dim, heads, ff_dim, dropout=dropout, sk_iters=sk_iters)
            for _ in range(layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, target_vocab_size)
        # Multi-task aux head (T1.2): word-segmentation boundary prediction.
        # Labels are trivially derived from input (1 at word boundaries, 0 elsewhere)
        # so no external labeler is needed — the value is encoder regularization.
        # POS head deferred to v2 (needs external POS tagger for labels).
        if with_seg_head:
            self.seg_head = nn.Linear(dim, 2)

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
        for layer in self.layers:
            x, prev_attn = layer(x, cos, sin, key_padding_mask, prev_attn)
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
        self.layers = nn.ModuleList([
            ModernEncoderLayer(dim, heads, ff_dim, dropout=dropout, sk_iters=sk_iters)
            for _ in range(layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.heads = nn.ModuleList([nn.Linear(dim, n) for n in head_sizes])

    def forward_encoder(self, src: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = src.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        key_padding_mask = src == self.pad_id
        x = self.embedding(src)
        cos, sin = self.rotary(seq_len)
        prev_attn: torch.Tensor | None = None
        for layer in self.layers:
            x, prev_attn = layer(x, cos, sin, key_padding_mask, prev_attn)
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
    )

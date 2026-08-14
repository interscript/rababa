"""ELECTRA-style pretraining (Replaced Token Detection).

Reference: Clark et al. ICLR 2020 (https://arxiv.org/abs/2003.10555).

Architecture: a small generator samples token replacements for masked
positions; a larger discriminator predicts per-position "is this token
the original?". The discriminator trains on ALL positions (vs MLM's
~15%) → ~2x sample efficiency.

For our char-level diacritization encoder:
  - Generator: small encoder + MLM head (samples replacements).
  - Discriminator: same arch as our supervised encoder + binary head.
  - After pretraining, discard generator. Discriminator's encoder IS
    our pretrain checkpoint.

Both share the input embedding (saves params + helps generator produce
realistic corruptions).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..constants import PAD_ID
from .optim import MuonAdamWHybrid
from .resume import latest_resume_checkpoint, save_resumable_checkpoint
from .supervised import TrainMetrics, build_optimizer, build_scheduler


@dataclass
class ElectraBatch:
    """Inputs + per-position original/replaced labels for the discriminator."""
    src: torch.Tensor           # (B, T) corrupted input IDs
    lengths: torch.Tensor       # (B,)
    replaced: torch.Tensor      # (B, T) int64 — 1 if replaced, 0 if original/PAD
    raw: list[str]


class ElectraDiscriminatorHead(nn.Module):
    """Binary per-position head: 'is this token original or replaced?'"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.act(self.dense(hidden))))


class ElectraModel(nn.Module):
    """Generator + Discriminator pair with shared embedding.

    The generator is a smaller encoder (e.g. hidden_dim/2, half layers).
    The discriminator is the same arch as our supervised encoder.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_mlm_head: nn.Module,
        discriminator_head: ElectraDiscriminatorHead,
        shared_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_mlm_head = generator_mlm_head
        self.discriminator_head = discriminator_head
        self.shared_embedding = shared_embedding
        dim = shared_embedding.embedding_dim
        self.gen_dim = dim
        self.disc_dim = discriminator.dim if hasattr(discriminator, "dim") else dim

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: generator samples replacements, discriminator predicts.

        Returns dict with keys: gen_logits, disc_logits, sampled_replacements.
        Caller computes losses from these + ground-truth `replaced` mask.
        """
        # Generator: predict original tokens at masked positions.
        gen_hidden = self.generator.forward_encoder(src)
        gen_logits = self.generator_mlm_head(gen_hidden)  # (B, T, V)

        # Sample replacements at masked positions.
        # For non-masked positions, keep original. For masked, sample from gen_logits.
        mask = self._sample_mask(src, mask_prob=0.15)
        sampled = gen_logits.argmax(dim=-1)
        # Use sampled only at mask positions; keep original elsewhere.
        corrupted = torch.where(mask, sampled, src)

        # Discriminator: predict per-position replaced/original.
        disc_hidden = self.discriminator.forward_encoder(corrupted)
        disc_logits = self.discriminator_head(disc_hidden)  # (B, T, 2)
        return {
            "gen_logits": gen_logits,
            "disc_logits": disc_logits,
            "corrupted": corrupted,
            "mask": mask,
        }

    @staticmethod
    def _sample_mask(src: torch.Tensor, mask_prob: float = 0.15) -> torch.Tensor:
        """Sample a boolean mask of positions to corrupt (excluding PAD)."""
        non_pad = src != PAD_ID
        rand = torch.rand_like(src, dtype=torch.float32)
        return (rand < mask_prob) & non_pad


def electra_loss(
    gen_logits: torch.Tensor,
    disc_logits: torch.Tensor,
    src: torch.Tensor,
    mask: torch.Tensor,
    corrupted: torch.Tensor,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute generator MLM loss + discriminator binary loss.

    Returns dict with 'total', 'gen', 'disc' keys.
    """
    # Generator: CE on masked positions (predict original).
    if mask.any():
        gen_targets = src[mask]
        gen_preds = gen_logits[mask]
        gen_loss = nn.functional.cross_entropy(
            gen_preds, gen_targets, label_smoothing=label_smoothing,
        )
    else:
        gen_loss = torch.tensor(0.0, device=src.device)

    # Discriminator: binary CE per non-PAD position.
    # Label: 1 if this position was replaced, 0 if kept original.
    non_pad = src != PAD_ID
    disc_targets = mask.long()
    # Flatten and apply non-pad mask.
    disc_logits_flat = disc_logits[non_pad]
    disc_targets_flat = disc_targets[non_pad]
    disc_loss = nn.functional.cross_entropy(disc_logits_flat, disc_targets_flat)
    # Weighted sum (ELECTRA paper recommends 50:1 disc:gen).
    total = gen_loss + 50.0 * disc_loss
    return {"total": total, "gen": gen_loss, "disc": disc_loss}


def build_electra_model(cfg: dict[str, Any]) -> ElectraModel:
    """Build an ElectraModel from config.

    Generator is half-width / half-layers of the discriminator.
    """
    from ..models.modern import ModernEncoderOnly, ModernEncoder
    from ..models.seq2seq import build_pretrain_model
    m = cfg.get("model", {})
    input_vocab = m.get("input_vocab_size", 100)
    dim = m.get("dim", 256)
    layers = m.get("layers", 6)
    heads = m.get("heads", 8)
    ff_dim = m.get("ff_dim", 1024)
    max_len = m.get("max_len", 128)

    # Discriminator: standard encoder-only.
    disc_encoder = build_pretrain_model(cfg)
    disc_dim = disc_encoder.encoder.dim

    # Generator: half the dim, half the layers. Same vocab embedding.
    gen_dim = max(dim // 2, 64)
    gen_layers = max(layers // 2, 1)
    gen_heads = max(heads // 2, 2)
    gen_ff_dim = max(ff_dim // 2, 128)
    generator = ModernEncoderOnly(
        vocab_size=input_vocab,
        dim=gen_dim,
        layers=gen_layers,
        heads=gen_heads,
        ff_dim=gen_ff_dim,
        max_len=max_len,
    )
    # Shared embedding: keep disc's embedding, point gen to it.
    # gen's embedding is dim=gen_dim, can't directly tie. So generator
    # uses a small projection from disc's embedding space.
    # For simplicity: don't tie; let generator have its own embedding.
    gen_mlm_head = nn.Linear(gen_dim, input_vocab, bias=False)
    gen_mlm_head.weight = generator.encoder.embedding.weight

    disc_head = ElectraDiscriminatorHead(disc_dim)
    return ElectraModel(
        generator=generator.encoder,  # Use the inner ModernEncoder for forward_encoder access
        discriminator=disc_encoder.encoder,
        generator_mlm_head=gen_mlm_head,
        discriminator_head=disc_head,
        shared_embedding=disc_encoder.encoder.embedding,
    )


def pretrain_electra(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root,
    log_fn=None,
) -> tuple[ElectraModel, any]:
    """Run ELECTRA pretraining. Returns (model, path to encoder checkpoint)."""
    from pathlib import Path
    ckpt_root = Path(ckpt_root)
    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 6)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    label_smoothing = cfg_train.get("label_smoothing", 0.0)

    model = build_electra_model(cfg).to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_root / "best.pt"

    resume = latest_resume_checkpoint(ckpt_root)
    if resume is not None:
        resume_path, last_epoch = resume
        if last_epoch >= 0:
            state = torch.load(resume_path, map_location=str(device), weights_only=False)
            if "model" in state:
                model.load_state_dict(state["model"])
            best_val = state.get("val_loss", float("inf"))
            start_epoch = last_epoch + 1

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                out = model(src, lengths)
                losses = electra_loss(
                    out["gen_logits"], out["disc_logits"], src,
                    out["mask"], out["corrupted"], label_smoothing,
                )
                loss = losses["total"]
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * src.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        # Validation: just compute total loss.
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch.src.to(device)
                lengths = batch.lengths.to(device)
                out = model(src, lengths)
                losses = electra_loss(
                    out["gen_logits"], out["disc_logits"], src,
                    out["mask"], out["corrupted"], 0.0,
                )
                val_loss += losses["total"].item() * src.size(0)
                n += src.size(0)
        val_loss /= max(1, n)

        if log_fn is not None:
            log_fn(TrainMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                learning_rate=optimizer.param_groups[0]["lr"]))

        full_ckpt_path = ckpt_root / f"checkpoint-epoch-{epoch}.pt"
        # Extract discriminator's encoder for fine-tune loading.
        encoder_state = model.discriminator.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "encoder_state_dict": encoder_state,
            },
            full_ckpt_path,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "val_loss": val_loss, "encoder_state_dict": encoder_state},
                best_path,
            )

    return model, best_path

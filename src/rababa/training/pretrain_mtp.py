"""Multi-Token Prediction (MTP) pretraining loop.

Reference: DeepSeek V4 (arXiv:2512.24880). Standard MLM predicts one
token per masked position. MTP predicts N tokens per position: the
current token + N-1 future tokens.

This module trains an encoder (same body used for supervised fine-tuning)
with an MTPHead on top: N parallel prediction heads with tied input
embedding. The encoder checkpoint is then loaded into the supervised
student with `strict=False`.

Dispatch path: `cfg.train.pretrain_method: "mtp"` in modal_app.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets import ArabicMLMDataset, MLMExample
from ..models.mlm import MLMModel, build_pretrain_model, extract_pretrained_encoder
from ..models.mtp import MTPHead, mtp_loss
from .collate import Batch
from .supervised import TrainMetrics, build_optimizer, build_scheduler


def mtp_collate_batch(batch: list[MLMExample], max_len: int = 200, n_predict: int = 2) -> Batch:
    """Pad MLM examples. Target is the unmasked input sequence so each
    head can pick off `target[:, i:i+T]`.

    The supervised Batch only carries `targets` as a list with a single
    tensor — for MTP we re-use that single tensor as the unmasked next-tokens
    buffer and let `mtp_loss` slice it per head.
    """
    truncated: list[MLMExample] = []
    for ex in batch:
        if len(ex.input_ids) > max_len:
            truncated.append(MLMExample(
                input_ids=ex.input_ids[:max_len],
                target_ids=ex.target_ids[:max_len],
                raw=ex.raw,
            ))
        else:
            truncated.append(ex)
    batch = truncated
    max_actual = max(len(ex.input_ids) for ex in batch)
    # Extend each sequence by n_predict-1 padding so mtp_loss can slice
    # target[:, i:i+T] for the last head without out-of-range.
    pad_target_len = max_actual + (n_predict - 1)
    src = torch.full((len(batch), max_actual), 0, dtype=torch.long)  # PAD_ID = 0
    target = torch.zeros((len(batch), pad_target_len), dtype=torch.long)
    lengths = torch.zeros((len(batch),), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        src[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        # For MTP, target_ids from MLMExample already store the unmasked
        # original token at each position. We copy them into target[i, :n]
        # so head i picks target[:, i:i+T] correctly.
        target[i, :n] = torch.tensor(ex.target_ids, dtype=torch.long)
        lengths[i] = n
    return Batch(src=src, lengths=lengths, targets=[target], raw=[ex.raw for ex in batch])


def make_mtp_collate_fn(max_len: int = 200, n_predict: int = 2):
    def _collate(batch: list[MLMExample]) -> Batch:
        return mtp_collate_batch(batch, max_len=max_len, n_predict=n_predict)
    return _collate


class MTPModel(nn.Module):
    """Encoder + MTPHead. Encoder weights are shared with the underlying model."""

    def __init__(self, encoder: nn.Module, n_predict: int = 2, tie_to_embedding: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        vocab_size = encoder.embedding.num_embeddings
        dim = encoder.embedding.embedding_dim
        self.head = MTPHead(
            dim=dim,
            vocab_size=vocab_size,
            n_predict=n_predict,
            tie_to_embedding=tie_to_embedding,
        )
        if tie_to_embedding:
            # Tie to encoder embedding for better generalization.
            self.head.shared_weight = encoder.embedding.weight

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.encoder.forward_encoder(src)
        return self.head(hidden)


def build_mtp_model(cfg: dict[str, Any]) -> MTPModel:
    """Build MTPModel from config. Reuses `build_pretrain_model` for the encoder."""
    from ..models.mlm import build_model
    encoder = build_model(cfg)
    n_predict = cfg.get("train", {}).get("mtp_n_predict", 2)
    return MTPModel(encoder, n_predict=n_predict, tie_to_embedding=True)


def evaluate_mtp(
    model: MTPModel,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            target = batch.targets[0].to(device)
            logits_list = model(src, lengths)
            loss = mtp_loss(logits_list, target, ignore_index=ignore_index)
            total_loss += loss.item() * src.size(0)
            total_count += src.size(0)
    return total_loss / max(1, total_count)


def pretrain_mtp(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    log_fn: Callable[[TrainMetrics], None] | None = None,
    metrics_path: Path | None = None,
) -> tuple[MTPModel, Path]:
    """Run MTP pretraining. Returns (model, path to encoder checkpoint).

    The encoder checkpoint contains embedding + transformer weights —
    loadable into a fresh student with `strict=False`. MTPHead weights
    are discarded (per DS4 spec — MTP is a pretrain-time-only objective).
    """
    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 3)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    n_predict = cfg_train.get("mtp_n_predict", 2)
    moe_lb_weight = cfg_train.get("moe_lb_weight", 0.01)

    from .resume import latest_resume_checkpoint
    from .metrics import MetricsLogger
    metrics_logger = MetricsLogger(metrics_path) if metrics_path is not None else None

    model = build_mtp_model(cfg).to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    def _collect_moe_lb() -> torch.Tensor:
        total = torch.tensor(0.0, device=device)
        for mod in model.modules():
            if hasattr(mod, "moe_load_balance_loss"):
                total = total + mod.moe_load_balance_loss()
        return total

    from .optim import MuonAdamWHybrid
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
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception:
                    pass
            best_val = state.get("val_loss", float("inf"))
            start_epoch = last_epoch + 1
            print(f"[resume] MTP continued from {resume_path.name} at epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            target = batch.targets[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                logits_list = model(src, lengths)
                loss = mtp_loss(logits_list, target, ignore_index=0)
                lb = _collect_moe_lb()
                if lb.requires_grad:
                    loss = loss + moe_lb_weight * lb
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                all_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                )
                if not all_finite:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * src.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss = evaluate_mtp(model, val_loader, device)
        metrics = TrainMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if log_fn is not None:
            log_fn(metrics)
        if metrics_logger is not None:
            metrics_logger.log(metrics)

        full_ckpt_path = ckpt_root / f"checkpoint-epoch-{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "encoder_state_dict": extract_pretrained_encoder_mtp(model),
            },
            full_ckpt_path,
        )
        val_is_better = (not math.isnan(val_loss)) and (val_loss < best_val)
        if val_is_better or (epoch == start_epoch and not best_path.is_file()):
            if val_is_better:
                best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "encoder_state_dict": extract_pretrained_encoder_mtp(model),
                },
                best_path,
            )

    if metrics_logger is not None:
        metrics_logger.close()
    return model, best_path


def extract_pretrained_encoder_mtp(mtp: MTPModel) -> dict[str, Any]:
    """Return encoder state_dict for fine-tune loading (excludes MTPHead)."""
    prefix = "encoder."
    skip_prefixes = ("encoder.head.", "encoder.heads.", "encoder.seg_head.")
    return {
        k[len(prefix):]: v
        for k, v in mtp.state_dict().items()
        if k.startswith(prefix) and not any(k.startswith(p) for p in skip_prefixes)
    }

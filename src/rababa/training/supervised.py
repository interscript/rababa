"""Unified Tier 1 supervised training loop — handles single- and multi-head models.

Both Arabic (1 head: haraqat) and Hebrew (3 heads: niqqud, dagesh, sin)
go through this same loop. The model exposes `forward_heads()` returning
a list of logits; the batch exposes `targets` as a list of per-head
ground truth. Loss = sum of per-head masked cross-entropies.

Pseudocode:
    for epoch in range(epochs):
        for batch in train_loader:
            outputs = model.forward_heads(batch.src, batch.lengths)
            loss = sum(CE(out, tgt) for out, tgt in zip(outputs, batch.targets))
            loss.backward(); optimizer.step(); scheduler.step()
        validate(model, val_loader)
        save_checkpoint(model, epoch)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..constants import PAD_ID
from ..models.base import build_model
from .collate import Batch

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adamw")
    lr = cfg.get("learning_rate", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        from .optim import MuonAdamWHybrid
        muon_lr = cfg.get("muon_lr", 0.02)
        return MuonAdamWHybrid(
            model,
            muon_lr=muon_lr,
            adam_lr=lr,
            adam_weight_decay=weight_decay,
            muon_momentum=cfg.get("muon_momentum", 0.95),
            ns_steps=cfg.get("ns_steps", 5),
        )
    raise ValueError(f"unknown optimizer: {name}")


def _lookup_space_id() -> int:
    """Char ID for space — used to derive segmentation labels from src."""
    from ..constants import VALID_ARABIC
    try:
        return VALID_ARABIC.index(" ") + 1
    except ValueError:
        return 0


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    name = cfg.get("scheduler", "cosine")
    warmup_steps = cfg.get("warmup_steps", 200)
    if name == "cosine":

        class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, warmup, total):
                self.warmup = warmup
                self.total = total
                super().__init__(optimizer)

            def get_lr(self):
                step = self.last_epoch
                if step < self.warmup:
                    return [base_lr * step / max(1, self.warmup) for base_lr in self.base_lrs]
                progress = (step - self.warmup) / max(1, self.total - self.warmup)
                return [
                    base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs
                ]

        return WarmupCosine(optimizer, warmup_steps, total_steps)
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    raise ValueError(f"unknown scheduler: {name}")


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross entropy ignoring PAD positions. Optional label smoothing."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    return nn.functional.cross_entropy(
        flat_logits,
        flat_target,
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
    )


def multi_head_loss(
    outputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    loss_fn: LossFn = masked_cross_entropy,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Sum of per-head losses. Outputs and targets must align by index.

    `label_smoothing` is forwarded to `loss_fn` if it accepts the kwarg;
    otherwise it's ignored (callers can also pre-bind via functools.partial).
    """
    if len(outputs) != len(targets):
        raise ValueError(
            f"head/output mismatch: {len(outputs)} outputs vs {len(targets)} targets"
        )
    total = 0
    for o, t in zip(outputs, targets, strict=True):
        try:
            total = total + loss_fn(o, t, label_smoothing=label_smoothing)
        except TypeError:
            # loss_fn doesn't accept label_smoothing; fall back to positional.
            total = total + loss_fn(o, t)
    return total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: LossFn = masked_cross_entropy,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            targets = [t.to(device) for t in batch.targets]
            outputs = model.forward_heads(src, lengths)
            loss = multi_head_loss(outputs, targets, loss_fn)
            total_loss += loss.item() * src.size(0)
            total_count += src.size(0)
    return total_loss / max(1, total_count)


def train_supervised(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    loss_fn: LossFn = masked_cross_entropy,
    log_fn: Callable[[TrainMetrics], None] | None = None,
) -> nn.Module:
    """Run supervised training. Returns the trained model.

    Checkpoints are written to `ckpt_root/checkpoint-epoch-{N}.pt`
    and `ckpt_root/best.pt` (lowest val_loss). Each checkpoint includes
    full optimizer + scheduler state so a Modal disconnect mid-run can
    be resumed by re-invoking this function — it auto-detects the
    latest checkpoint and continues from the next epoch.
    """
    from .resume import (
        latest_resume_checkpoint,
        load_resume_state,
        save_resumable_checkpoint,
    )

    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 5)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    label_smoothing = cfg_train.get("label_smoothing", 0.0)
    init_from_pretrain = cfg_train.get("init_from_pretrain")

    model = build_model(cfg).to(device)
    if init_from_pretrain:
        from .pretrain import load_pretrained_encoder
        load_pretrained_encoder(Path(init_from_pretrain), model)
        model.to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    best_val = float("inf")
    start_epoch = 0
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Resume from latest checkpoint if one exists (Modal disconnect recovery).
    resume = latest_resume_checkpoint(ckpt_root)
    if resume is not None:
        resume_path, last_epoch = resume
        if last_epoch >= 0:
            state = load_resume_state(model, optimizer, scheduler, resume_path, device=str(device))
            best_val = state.get("best_val_loss", float("inf"))
            start_epoch = last_epoch + 1
            if log_fn is not None:
                log_fn(TrainMetrics(
                    epoch=last_epoch, train_loss=0.0, val_loss=best_val,
                    learning_rate=optimizer.param_groups[0]["lr"],
                ))
            print(f"[resume] continued from {resume_path.name} at epoch {start_epoch}/{epochs}")

    def _loss_fn(logits, target, label_smoothing=0.0):
        return loss_fn(logits, target, label_smoothing=label_smoothing)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        # Detect multi-task model (e.g., ModernCharTransformer with seg head).
        head_names = model.head_names() if hasattr(model, "head_names") else ["output"]
        has_seg = "seg" in head_names
        space_id = _lookup_space_id()
        for batch in train_loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            targets = [t.to(device) for t in batch.targets]
            # Generate segmentation labels on-the-fly from src if the model
            # exposes a seg head. Label = 1 at the first char of each word.
            if has_seg and len(targets) < len(head_names):
                seg = torch.zeros_like(src)
                seg[:, 0] = 1  # first char of sequence starts a word
                # Position after a space starts a new word.
                seg[:, 1:] = (src[:, :-1] == space_id).long()
                targets = targets + [seg]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                outputs = model.forward_heads(src, lengths)
                loss = multi_head_loss(outputs, targets, _loss_fn, label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * src.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss = evaluate(model, val_loader, device, _loss_fn)
        metrics = TrainMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if log_fn is not None:
            log_fn(metrics)

        # Save resumable checkpoint with full state.
        save_resumable_checkpoint(
            ckpt_root / f"checkpoint-epoch-{epoch}.pt",
            model, optimizer, scheduler,
            epoch=epoch, best_val_loss=best_val,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_root / "best.pt")

    return model

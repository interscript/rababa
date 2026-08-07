"""MLM pretraining loop — mirrors train_supervised structure.

Trains a CharTransformer + MLMHead with masked-LM cross-entropy. The
encoder weights from the trained MLMModel are saved separately and
loaded into a fresh student (with `strict=False`) for Tier 1 supervised
fine-tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets import ArabicMLMDataset, MLMExample
from ..models.mlm import MLMModel, build_pretrain_model, extract_pretrained_encoder
from .collate import Batch
from .supervised import TrainMetrics, build_optimizer, build_scheduler, masked_cross_entropy


def mlm_collate_batch(batch: list[MLMExample], max_len: int = 200) -> Batch:
    """Pad MLM examples. Same shape as supervised Batch (different source type)."""
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
    src = torch.full((len(batch), max_actual), 0, dtype=torch.long)  # PAD_ID = 0
    target = torch.zeros((len(batch), max_actual), dtype=torch.long)
    lengths = torch.zeros((len(batch),), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        src[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        target[i, :n] = torch.tensor(ex.target_ids, dtype=torch.long)
        lengths[i] = n
    return Batch(src=src, lengths=lengths, targets=[target], raw=[ex.raw for ex in batch])


def make_mlm_collate_fn(max_len: int = 200):
    def _collate(batch: list[MLMExample]) -> Batch:
        return mlm_collate_batch(batch, max_len=max_len)
    return _collate


def evaluate_mlm(
    model: MLMModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            target = batch.targets[0].to(device)
            logits = model(src, lengths)
            total_loss += masked_cross_entropy(logits, target).item() * src.size(0)
            total_count += src.size(0)
    return total_loss / max(1, total_count)


def pretrain_mlm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    log_fn: Callable[[TrainMetrics], None] | None = None,
) -> tuple[MLMModel, Path]:
    """Run MLM pretraining. Returns (model, path to encoder checkpoint).

    The encoder checkpoint contains the embedding, position, and transformer
    weights — loadable into a fresh CharTransformer with `strict=False`.
    """
    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 3)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)

    from .resume import latest_resume_checkpoint

    model = build_pretrain_model(cfg).to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    from .optim import MuonAdamWHybrid
    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_root / "best.pt"

    # Resume from latest checkpoint if one exists (Modal disconnect recovery).
    resume = latest_resume_checkpoint(ckpt_root)
    if resume is not None:
        resume_path, last_epoch = resume
        if last_epoch >= 0:
            state = torch.load(resume_path, map_location=str(device), weights_only=False)
            model.load_state_dict(state["model"]) if "model" in state else None
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception:
                    pass
            best_val = state.get("val_loss", float("inf"))
            start_epoch = last_epoch + 1
            print(f"[resume] continued from {resume_path.name} at epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            target = batch.targets[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                logits = model(src, lengths)
                loss = masked_cross_entropy(logits, target)
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
        val_loss = evaluate_mlm(model, val_loader, device)
        metrics = TrainMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if log_fn is not None:
            log_fn(metrics)

        # Save encoder checkpoint with full resume state.
        full_ckpt_path = ckpt_root / f"checkpoint-epoch-{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "encoder_state_dict": extract_pretrained_encoder(model),
            },
            full_ckpt_path,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "encoder_state_dict": extract_pretrained_encoder(model),
                },
                best_path,
            )

    return model, best_path


def load_pretrained_encoder(checkpoint_path: Path, model: nn.Module) -> None:
    """Load encoder weights from a pretrain checkpoint into a fresh student.

    Loads with `strict=False` so the haraqat head is left at its init.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = ckpt["encoder_state_dict"] if "encoder_state_dict" in ckpt else ckpt
    model.load_state_dict(encoder_state, strict=False)

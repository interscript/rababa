"""Distillation — train a student from N teacher checkpoints.

Soft-label KL divergence against the ensemble's averaged probabilities,
mixed with hard CE against gold labels. α schedule anneals from
teacher-guided to gold-only across training.

Reuses `train_supervised` infrastructure: same optimizer, scheduler,
checkpoint resume, log_fn. The only addition is the teacher inference
pass per batch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..constants import PAD_ID
from ..models.base import build_model
from .collate import Batch
from .multi_seed import teacher_checkpoint_paths
from .resume import (
    latest_resume_checkpoint,
    load_resume_state,
    save_resumable_checkpoint,
)
from .supervised import (
    TrainMetrics,
    _lookup_space_id,
    build_optimizer,
    build_scheduler,
    evaluate as eval_supervised,
    masked_cross_entropy,
    multi_head_loss,
)


def load_teachers(
    teacher_paths: list[Path],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[nn.Module]:
    """Load N teacher models, all sharing the same architecture."""
    teachers: list[nn.Module] = []
    for p in teacher_paths:
        m = build_model(cfg).to(device).eval()
        state = torch.load(p, map_location=device, weights_only=True)
        # Tolerate both wrapped and raw state_dict (resume.py compat).
        if "model" in state:
            state = state["model"]
        m.load_state_dict(state)
        for param in m.parameters():
            param.requires_grad_(False)
        teachers.append(m)
    return teachers


def averaged_teacher_logits(
    teachers: list[nn.Module],
    src: torch.Tensor,
    lengths: torch.Tensor,
) -> list[torch.Tensor]:
    """Run each teacher, return per-head averaged logits.

    Returns: list of (B, T, V) tensors, one per head, in canonical order.
    """
    head_outputs: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for teacher in teachers:
            outs = teacher.forward_heads(src, lengths)
            if not head_outputs:
                head_outputs = [[o] for o in outs]
            else:
                for i, o in enumerate(outs):
                    head_outputs[i].append(o)
    return [torch.stack(head_list).mean(dim=0) for head_list in head_outputs]


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    temperature: float = 2.0,
) -> torch.Tensor:
    """(1-α)·CE(student, gold) + α·KL(student, teacher_avg).

    Args:
        student_logits, teacher_logits: (B, T, V)
        target: (B, T) int64 with PAD_ID = ignore.
        alpha: 0.0 = pure gold, 1.0 = pure teacher.
        temperature: softmax temperature for KL (lower = sharper).
    """
    ce = masked_cross_entropy(student_logits, target, label_smoothing=0.0)
    # KL: softmax(teacher/T) · log(softmax(student/T) / softmax(teacher/T))
    log_p_student = torch.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    kl_per_pos = (p_teacher * (torch.log_softmax(teacher_logits / temperature, dim=-1) - log_p_student)).sum(-1)
    mask = (target != PAD_ID).float()
    kl = (kl_per_pos * mask).sum() / mask.sum().clamp_min(1.0) * (temperature ** 2)
    return (1 - alpha) * ce + alpha * kl


def distill_into_student(
    teachers: list[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    alpha_init: float = 0.5,
    alpha_final: float = 0.0,
    temperature: float = 2.0,
    log_fn: Callable[[TrainMetrics], None] | None = None,
) -> nn.Module:
    """Train a student against teacher-averaged soft labels + gold hard labels.

    Student's architecture matches teachers (built via `build_model(cfg)`).
    Teacher weights are frozen.

    α schedule is linear from `alpha_init` → `alpha_final` across epochs.
    """
    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 20)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    label_smoothing = cfg_train.get("label_smoothing", 0.1)

    student = build_model(cfg).to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(student, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    from .optim import MuonAdamWHybrid
    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
    ckpt_root.mkdir(parents=True, exist_ok=True)

    resume = latest_resume_checkpoint(ckpt_root)
    if resume is not None:
        resume_path, last_epoch = resume
        if last_epoch >= 0:
            state = load_resume_state(student, optimizer, scheduler, resume_path, device=str(device))
            best_val = state.get("best_val_loss", float("inf"))
            start_epoch = last_epoch + 1

    def _loss_fn(logits, target, label_smoothing=0.0):
        return masked_cross_entropy(logits, target, label_smoothing=label_smoothing)

    for epoch in range(start_epoch, epochs):
        # Linear alpha anneal.
        if epochs > 1:
            alpha = alpha_init + (alpha_final - alpha_init) * (epoch / max(1, epochs - 1))
        else:
            alpha = alpha_final
        student.train()
        running_loss = 0.0
        head_names = student.head_names() if hasattr(student, "head_names") else ["output"]
        has_seg = "seg" in head_names
        space_id = _lookup_space_id()
        for batch in train_loader:
            src = batch.src.to(device)
            lengths = batch.lengths.to(device)
            targets = [t.to(device) for t in batch.targets]
            if has_seg and len(targets) < len(head_names):
                seg = torch.zeros_like(src)
                seg[:, 0] = 1
                seg[:, 1:] = (src[:, :-1] == space_id).long()
                targets = targets + [seg]
            # Teacher inference (no grad).
            teacher_heads = averaged_teacher_logits(teachers, src, lengths)
            # Pad teacher_heads to match head count (seg head has no teacher).
            while len(teacher_heads) < len(targets):
                teacher_heads.append(torch.zeros_like(targets[-1].unsqueeze(-1).expand(-1, -1, 1).float()))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                outputs = student.forward_heads(src, lengths)
                loss = sum(
                    distillation_loss(s, t, tgt, alpha=alpha, temperature=temperature)
                    for s, t, tgt in zip(outputs, teacher_heads, targets, strict=True)
                )
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * src.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss = eval_supervised(student, val_loader, device, _loss_fn)
        if log_fn is not None:
            log_fn(TrainMetrics(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
            ))
        save_resumable_checkpoint(
            ckpt_root / f"checkpoint-epoch-{epoch}.pt",
            student, optimizer, scheduler,
            epoch=epoch, best_val_loss=best_val,
            extra={"stage": "distill"},
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), ckpt_root / "best.pt")
    return student


def distill_from_checkpoints(
    teacher_paths: list[Path],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    **kwargs,
) -> nn.Module:
    """Convenience: load teachers from paths then call distill_into_student."""
    teachers = load_teachers(teacher_paths, cfg, device)
    return distill_into_student(
        teachers, train_loader, val_loader, cfg, device, ckpt_root, **kwargs,
    )

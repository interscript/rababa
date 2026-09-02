"""Multi-seed training — train N copies of a task with different seeds.

Each seed runs as an independent Modal function call (parallel via
`.starmap()`). Each writes to `/checkpoints/{task}/run-{seed:03d}/`
so the distillation stage can find all teachers.

Single source of truth: `train_with_seed(task, seed)` runs one full
train cycle. The orchestrator (`scripts/train_seeds.py`) is a thin
wrapper that dispatches N calls in parallel.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from .collate import Batch
from .supervised import TrainMetrics, build_optimizer, build_scheduler, masked_cross_entropy, multi_head_loss
from ..models.base import build_model
from ..tasks import build_supervised_loaders
from ..config import load_task_config, to_dict
from .resume import latest_resume_checkpoint, save_resumable_checkpoint, load_resume_state


def _set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_with_seed(
    task: str,
    seed: int,
    ckpt_root: Path,
    cfg_dict: dict | None = None,
    device: torch.device | None = None,
    log_fn=None,
) -> Path:
    """Train one model with a specific seed. Returns path to best.pt.

    Args:
        task: task name (e.g. rababa_arabic_pro).
        seed: random seed for model init + data shuffle.
        ckpt_root: directory to write checkpoints. Should be unique per
            seed to avoid collision (typically /checkpoints/{task}/run-{seed:03d}).
        cfg_dict: optional pre-loaded config dict. If None, loads from disk.
        device: torch device. Defaults to CUDA if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg_dict is None:
        cfg = load_task_config(task)
        cfg_dict = to_dict(cfg)
    else:
        cfg = type("Cfg", (), cfg_dict)  # type: ignore[assignment]

    _set_seed(seed)
    cfg_train = cfg_dict.get("train", {})
    epochs = cfg_train.get("epochs", 20)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    label_smoothing = cfg_train.get("label_smoothing", 0.1)
    init_from_pretrain = cfg_train.get("init_from_pretrain")

    # Re-load config here so the supervisor API matches train_supervised.
    from ..config import load_task_config, to_dict
    cfg_omega = load_task_config(task)
    train_loader, val_loader = build_supervised_loaders(cfg_omega)

    model = build_model(cfg_dict).to(device)
    if init_from_pretrain:
        from .pretrain import load_pretrained_encoder
        load_pretrained_encoder(Path(init_from_pretrain), model)
        model.to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
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
            state = load_resume_state(model, optimizer, scheduler, resume_path, device=str(device))
            best_val = state.get("best_val_loss", float("inf"))
            start_epoch = last_epoch + 1

    def _loss_fn(logits, target, label_smoothing=0.0):
        return masked_cross_entropy(logits, target, label_smoothing=label_smoothing)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        head_names = model.head_names() if hasattr(model, "head_names") else ["output"]
        has_seg = "seg" in head_names
        from .supervised import _lookup_space_id
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
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                outputs = model.forward_heads(src, lengths)
                loss = multi_head_loss(outputs, targets, _loss_fn, label_smoothing)
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

        from .supervised import evaluate
        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss = evaluate(model, val_loader, device, _loss_fn)
        if log_fn is not None:
            log_fn(TrainMetrics(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
            ))
        save_resumable_checkpoint(
            ckpt_root / f"checkpoint-epoch-{epoch}.pt",
            model, optimizer, scheduler,
            epoch=epoch, best_val_loss=best_val,
            extra={"seed": seed},
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_root / "best.pt")

    return ckpt_root / "best.pt"


def teacher_checkpoint_paths(task: str, n_seeds: int, root: str = "/checkpoints") -> list[Path]:
    """Return list of `best.pt` paths for a multi-seed ensemble.

    Looks for `{root}/{task}/seed-{NNN}/run-001/best.pt` where NNN goes
    from 0 to n_seeds-1. Missing files are silently skipped — caller can
    detect partial ensembles by checking len(result) < n_seeds.
    """
    out: list[Path] = []
    base = Path(root) / task
    for seed in range(n_seeds):
        p = base / f"seed-{seed:03d}" / "run-001" / "best.pt"
        if p.is_file():
            out.append(p)
    return out

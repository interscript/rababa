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
            cross_attn_lr_mult=cfg.get("cross_attn_lr_mult", 1.0),
            spectral_cap=cfg.get("spectral_cap"),
            heavy_tail_alpha=cfg.get("heavy_tail_alpha"),
            adamuon_beta=cfg.get("adamuon_beta"),
            normuon_enabled=cfg.get("normuon_enabled", False),
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
    optimizer: Any,
    cfg: dict[str, Any],
    total_steps: int,
) -> Any:
    """Build LR scheduler. Accepts torch.optim.Optimizer OR MuonAdamWHybrid.

    WarmupCosine is a plain object (not LRScheduler subclass) so it works
    with MuonAdamWHybrid, which is not a torch.optim.Optimizer.
    """
    name = cfg.get("scheduler", "cosine")
    warmup_steps = cfg.get("warmup_steps", 200)
    if name == "cosine":

        class WarmupCosine:
            """Cosine decay with linear warmup. Duck-types LRScheduler."""

            def __init__(self, optimizer, warmup, total):
                self.optimizer = optimizer
                self.warmup = warmup
                self.total = total
                self.last_epoch = 0
                self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

            def get_lr(self) -> list[float]:
                step = self.last_epoch
                if step < self.warmup:
                    return [base * step / max(1, self.warmup) for base in self.base_lrs]
                progress = (step - self.warmup) / max(1, self.total - self.warmup)
                return [base * 0.5 * (1 + math.cos(math.pi * progress)) for base in self.base_lrs]

            def step(self) -> None:
                self.last_epoch += 1
                for g, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    g["lr"] = lr

            def state_dict(self) -> dict[str, Any]:
                return {"last_epoch": self.last_epoch, "base_lrs": list(self.base_lrs)}

            def load_state_dict(self, state: dict[str, Any]) -> None:
                self.last_epoch = int(state.get("last_epoch", 0))
                if "base_lrs" in state:
                    self.base_lrs = list(state["base_lrs"])

        return WarmupCosine(optimizer, warmup_steps, total_steps)
    if name == "constant":
        if not isinstance(optimizer, torch.optim.Optimizer):
            # No-op scheduler for hybrid optimizers under constant schedule.
            class _NoOp:
                def step(self): pass
                def state_dict(self): return {}
                def load_state_dict(self, s): pass
            return _NoOp()
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    raise ValueError(f"unknown scheduler: {name}")


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Cross entropy ignoring PAD positions. Optional label smoothing.

    Args:
        class_weights: per-class weights (shape [n_classes]). When provided,
          up-weights rare classes. Use `compute_class_weights()` to build
          inverse-frequency weights from training data.
        focal_gamma: focal loss gamma. >0 reduces loss for well-classified
          examples (puts more emphasis on hard examples). 0 = standard CE.
          Typical: 1.0-2.0 for imbalanced tasks.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    if focal_gamma and focal_gamma > 0:
        # Focal loss: (1 - p_t)^γ * CE. Reduces contribution of easy examples.
        ce = nn.functional.cross_entropy(
            flat_logits, flat_target,
            ignore_index=PAD_ID,
            label_smoothing=label_smoothing,
            weight=class_weights,
            reduction="none",
        )
        with torch.no_grad():
            p_t = torch.gather(flat_logits.softmax(-1), -1, flat_target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            p_t = p_t.where(flat_target != PAD_ID, torch.ones_like(p_t))
        loss = ((1 - p_t) ** focal_gamma) * ce
        # Mean over non-pad positions only.
        mask = flat_target != PAD_ID
        return loss.sum() / mask.sum().clamp_min(1)
    return nn.functional.cross_entropy(
        flat_logits,
        flat_target,
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
        weight=class_weights,
    )


def entropy_regularizer(logits: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
    """Entropy regularization loss: -weight * E[H(p)].

    Encourages the model to be less confident (higher entropy) — prevents
    overconfident wrong predictions on rare classes. Adds H(p) to the loss
    with negative sign so minimizing loss maximizes entropy.

    Args:
        logits: (B, T, V) or (N, V) logits.
        weight: regularization weight. 0 = disabled. 0.01-0.1 typical.

    Returns: scalar loss (0 if weight=0).
    """
    if weight <= 0:
        return torch.tensor(0.0, device=logits.device)
    flat = logits.reshape(-1, logits.size(-1))
    probs = flat.softmax(-1)
    log_probs = flat.log_softmax(-1)
    entropy = -(probs * log_probs).sum(-1).mean()
    # We want to MAXIMIZE entropy, so add negative as loss.
    return -weight * entropy


def compute_class_weights(
    targets: list[torch.Tensor],
    n_classes: list[int],
    smoothing: float = 0.1,
) -> list[torch.Tensor]:
    """Compute inverse-frequency class weights from training targets.

    For each head, count class frequencies, then weights = (1 / freq) normalized.
    Smoothing prevents division by zero and extreme weights for very rare classes.
    """
    weights = []
    for head_targets, n_cls in zip(targets, n_classes):
        flat = head_targets.reshape(-1)
        flat = flat[flat != PAD_ID]
        if flat.numel() == 0:
            weights.append(torch.ones(n_cls))
            continue
        counts = torch.bincount(flat, minlength=n_cls).float()
        counts = counts + smoothing * counts.max()  # smooth
        w = counts.sum() / (n_cls * counts)
        w = w / w.mean()  # normalize to mean=1
        weights.append(w)
    return weights


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
            # Seq2seq path: teacher-forced CE on decoder output.
            if hasattr(batch, "tgt_in"):
                from ..constants import PAD_ID as _PID
                tgt_in = batch.tgt_in.to(device)
                tgt_out = batch.tgt_out.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                    logits = model.forward(src, tgt_in)
                    vocab_size = logits.size(-1)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        tgt_out.reshape(-1),
                        ignore_index=_PID,
                    )
                total_loss += loss.item() * src.size(0)
                total_count += src.size(0)
                continue
            targets = [t.to(device) for t in batch.targets]
            outputs = model.forward_heads(src, lengths)
            # Use plain CE (no class weights, no focal) for evaluation so
            # val_loss is comparable across configs and isn't biased by
            # the training-only class weighting.
            total = 0
            for o, t in zip(outputs, targets, strict=True):
                total = total + masked_cross_entropy(o, t)
            loss = total
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
    metrics_path: Path | None = None,
) -> nn.Module:
    """Run supervised training. Returns the trained model.

    Checkpoints are written to `ckpt_root/checkpoint-epoch-{N}.pt`
    and `ckpt_root/best.pt` (lowest val_loss). Each checkpoint includes
    full optimizer + scheduler state so a Modal disconnect mid-run can
    be resumed by re-invoking this function — it auto-detects the
    latest checkpoint and continues from the next epoch.

    Args:
        metrics_path: if provided, per-epoch metrics (train_loss, val_loss,
            learning_rate) are appended to this JSONL file via MetricsLogger.
    """
    from .resume import (
        latest_resume_checkpoint,
        load_resume_state,
        save_resumable_checkpoint,
    )

    metrics_logger = None
    if metrics_path is not None:
        from .metrics import MetricsLogger
        metrics_logger = MetricsLogger(metrics_path)

    cfg_train = cfg.get("train", {})

    # Optional EMA (Exponential Moving Average) of model weights.
    # Smooths predictions and improves generalization by 2-5% on NLP tasks.
    ema_decay = float(cfg_train.get("ema_decay", 0.0))
    ema = None  # initialized AFTER model is created (need params).
    epochs = cfg_train.get("epochs", 5)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    label_smoothing = cfg_train.get("label_smoothing", 0.0)
    init_from_pretrain = cfg_train.get("init_from_pretrain")
    focal_gamma = float(cfg_train.get("focal_gamma", 0.0))
    use_class_weights = bool(cfg_train.get("class_weights", False))
    entropy_weight = float(cfg_train.get("entropy_weight", 0.0))

    # Optional curriculum learning: order training examples by difficulty,
    # expose harder examples as training progresses.
    cur_cfg = cfg_train.get("curriculum", {}) or {}
    if cur_cfg.get("enabled", False):
        from .curriculum import CurriculumSampler
        try:
            from ..features.arabic import compute_arabic_features
            def _difficulty(ex):
                feats = compute_arabic_features(ex)
                return feats.get("iltiqaa_violation", 0) + feats.get("word_boundary", 0) * 0.1
        except ImportError:
            def _difficulty(ex):
                return 0
        cur_sampler = CurriculumSampler(
            dataset=train_loader.dataset,
            difficulty_fn=_difficulty,
            n_buckets=int(cur_cfg.get("n_buckets", 5)),
            total_epochs=epochs,
            schedule=str(cur_cfg.get("schedule", "linear")),
        )
        from torch.utils.data import DataLoader as _DL
        train_loader = _DL(
            train_loader.dataset,
            sampler=cur_sampler,
            batch_size=int(cfg_train.get("batch_size", 32)),
            num_workers=int(cfg_train.get("num_workers", 8)),
            collate_fn=train_loader.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    model = build_model(cfg).to(device)
    if init_from_pretrain:
        from .pretrain import load_pretrained_encoder
        load_pretrained_encoder(Path(init_from_pretrain), model)
        model.to(device)
    # Initialize EMA after model load (so it tracks pretrained weights too).
    if ema_decay > 0:
        from .ema import ModelEMA
        ema = ModelEMA(model, decay=ema_decay)
        print(f"[train] EMA enabled, decay={ema_decay}", flush=True)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    # Muon hybrid is not a torch.optim.Optimizer — GradScaler can't step it.
    # bf16 autocast has enough dynamic range that GradScaler is unnecessary.
    from .optim import MuonAdamWHybrid
    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
    epochs_since_best = 0
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
            print(f"[resume] continued from {resume_path.name} at epoch {start_epoch}/{epochs}", flush=True)
        else:
            print(f"[resume] found {resume_path.name} but last_epoch={last_epoch}, starting fresh", flush=True)
    else:
        print(f"[train] starting fresh from epoch 0/{epochs}, ckpt_root={ckpt_root}", flush=True)

    print(f"[train] train_loader batches={len(train_loader)}, dataset={len(train_loader.dataset)}", flush=True)
    print(f"[train] val_loader batches={len(val_loader)}, dataset={len(val_loader.dataset)}", flush=True)

    # Compute per-head class weights from training data (one-shot, before training).
    head_class_weights: list[torch.Tensor] | None = None
    if use_class_weights:
        head_names = model.head_names() if hasattr(model, "head_names") else ["output"]
        # Get head sizes from the model's output heads (more reliable than data max).
        head_sizes: list[int] = []
        if hasattr(model, "heads") and isinstance(model.heads, nn.ModuleList):
            for h in model.heads:
                head_sizes.append(h.out_features)
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            head_sizes.append(model.head.out_features)
        else:
            # Fallback: derive from data max.
            for _ in head_names:
                head_sizes.append(0)
        # Gather all targets across training set.
        all_targets_per_head: list[list[torch.Tensor]] = [[] for _ in head_names]
        for batch in train_loader:
            for h_idx, t in enumerate(batch.targets):
                if h_idx < len(all_targets_per_head):
                    all_targets_per_head[h_idx].append(t.clone())
        # Compute weights per head using MODEL head size (data max may be smaller).
        head_class_weights = []
        for h_idx, target_list in enumerate(all_targets_per_head):
            if not target_list:
                head_class_weights.append(None)
                continue
            flat = torch.cat([t.reshape(-1) for t in target_list])
            n_cls = head_sizes[h_idx] if head_sizes and h_idx < len(head_sizes) else int(flat.max().item()) + 1
            w = compute_class_weights([flat], [n_cls])[0]
            head_class_weights.append(w.to(device))
        head_class_weights = [w for w in head_class_weights if w is not None]
        print(f"[train] computed class weights for {len(head_class_weights)} heads", flush=True)
        if head_class_weights:
            for i, w in enumerate(head_class_weights):
                print(f"  head {i} (n={w.shape[0]}): min={w.min().item():.3f} max={w.max().item():.3f} mean={w.mean().item():.3f}", flush=True)

    def _loss_fn(logits, target, label_smoothing=0.0, head_idx=0):
        cw = head_class_weights[head_idx] if head_class_weights and head_idx < len(head_class_weights) else None
        return masked_cross_entropy(logits, target, label_smoothing=label_smoothing,
                                    class_weights=cw, focal_gamma=focal_gamma)

    moe_lb_weight = cfg_train.get("moe_lb_weight", 0.01)

    def _collect_moe_lb() -> torch.Tensor:
        """Sum load-balance losses from every LatentMoE submodule.

        Without this, fine-grained MoE collapses to a single expert.
        Qwen3 uses global-batch normalization (default in LatentMoE).
        """
        total = torch.tensor(0.0, device=device)
        for mod in model.modules():
            if hasattr(mod, "moe_load_balance_loss"):
                total = total + mod.moe_load_balance_loss()
        return total

    # Optional NaN auto-recovery: detect divergence, restore last good state,
    # halve LR, resume. Disabled when cfg.train.nan_recovery is False.
    recovery = None
    if cfg_train.get("nan_recovery", True):
        from .recovery import NaNAutoRecovery
        recovery = NaNAutoRecovery(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_root=ckpt_root,
            max_recoveries=int(cfg_train.get("nan_recovery_max", 3)),
            lr_scale=float(cfg_train.get("nan_recovery_lr_scale", 0.5)),
        )

    # Outer loop allows recovery to restart from a saved epoch.
    _recovered = False
    while True:
        _recovered = False
        # Reset early-stopping counter on each (re)start so NaN recovery
        # doesn't carry over stale patience debt.
        epochs_since_best = 0

        for epoch in range(start_epoch, epochs):
            print(f"[train] starting epoch {epoch}/{epochs}, scheduler_step={scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else '?'}", flush=True)
            # Update curriculum sampler's epoch pointer if active.
            if cur_cfg.get("enabled", False) and hasattr(train_loader, "sampler"):
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            head_names = model.head_names() if hasattr(model, "head_names") else ["output"]
            has_seg = "seg" in head_names
            space_id = _lookup_space_id()
            for batch in train_loader:
                src = batch.src.to(device)
                lengths = batch.lengths.to(device)
                optimizer.zero_grad(set_to_none=True)
                # Seq2seq path: teacher-forced CE on decoder output.
                if hasattr(batch, 'tgt_in'):
                    from ..constants import PAD_ID as _PID
                    tgt_in = batch.tgt_in.to(device)
                    tgt_out = batch.tgt_out.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                        logits = model.forward(src, tgt_in)
                        vocab_size = logits.size(-1)
                        loss = nn.functional.cross_entropy(
                            logits.reshape(-1, vocab_size),
                            tgt_out.reshape(-1),
                            ignore_index=_PID,
                            label_smoothing=label_smoothing,
                        )
                    if not torch.isfinite(loss):
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    # Guard against NaN/Inf gradients (common in bf16 seq2seq).
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
                    continue
                # Classification path (existing code).
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
                    if head_class_weights is not None:
                        # Per-head weighted loss: bypass multi_head_loss so we can
                        # pass head_idx for per-head class_weights + focal_gamma.
                        loss = 0
                        for h_idx, (o, t) in enumerate(zip(outputs, targets, strict=True)):
                            loss = loss + _loss_fn(o, t, label_smoothing=label_smoothing, head_idx=h_idx)
                    else:
                        loss = multi_head_loss(outputs, targets, _loss_fn, label_smoothing)
                    # Optional entropy regularization (prevents overconfidence).
                    if entropy_weight > 0:
                        for o in outputs:
                            loss = loss + entropy_regularizer(o, weight=entropy_weight)
                    lb = _collect_moe_lb()
                    if lb.requires_grad:
                        loss = loss + moe_lb_weight * lb
                # Skip NaN/Inf loss — protects weights from poisoning.
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
                # Update EMA shadow weights after each step.
                if ema is not None:
                    ema.update(model)
                scheduler.step()
                running_loss += loss.item() * src.size(0)

            train_loss = running_loss / max(1, len(train_loader.dataset))
            # Use EMA copy for evaluation if available (smoothed predictions).
            if ema is not None:
                with ema.swap(model):
                    val_loss = evaluate(model, val_loader, device, _loss_fn)
            else:
                val_loss = evaluate(model, val_loader, device, _loss_fn)
            print(
                f"[train] epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
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

            # NaN auto-recovery: snapshot good state, or trigger recovery on NaN.
            if recovery is not None:
                if not math.isnan(val_loss):
                    recovery.checkpoint_good(epoch, val_loss)
                elif recovery.can_recover():
                    start_epoch = recovery.recover()
                    _recovered = True
                    break  # restart outer loop at start_epoch
                # else: max recoveries exhausted — keep going, save what we have

            # Save resumable checkpoint with full state.
            save_resumable_checkpoint(
                ckpt_root / f"checkpoint-epoch-{epoch}.pt",
                model, optimizer, scheduler,
                epoch=epoch, best_val_loss=best_val,
            )
            # Update best.pt. Skip NaN val_loss (numerical instability) — never
            # let NaN be "best". But always save best.pt on the first epoch if
            # it doesn't exist yet, so downstream stages can find a checkpoint.
            best_path = ckpt_root / "best.pt"
            val_is_better = (not math.isnan(val_loss)) and (val_loss < best_val)
            if val_is_better or (epoch == start_epoch and not best_path.is_file()):
                if val_is_better:
                    best_val = val_loss
                    epochs_since_best = 0
                # If EMA is enabled, save EMA weights as best.pt (better
                # generalization at inference than live weights).
                if ema is not None:
                    with ema.swap(model):
                        torch.save(model.state_dict(), best_path)
                else:
                    torch.save(model.state_dict(), best_path)
            else:
                epochs_since_best += 1

            # Early stopping: break if val_loss hasn't improved for `patience`
            # epochs. Skipped during NaN recovery restarts (start_epoch resets).
            es_patience = int(cfg_train.get("early_stopping_patience", 0))
            if (
                es_patience > 0
                and epochs_since_best >= es_patience
                and epoch < epochs - 1  # don't double-break on last epoch
            ):
                print(
                    f"[train] early stopping at epoch {epoch}: no val_loss improvement "
                    f"for {epochs_since_best} epochs (patience={es_patience})",
                    flush=True,
                )
                break  # break for-loop; exit while via _done flag below.

            # End of for-loop body. If we got here normally (no recovery break),
            # we're done with all epochs — exit outer while.
            if not _recovered and epoch == epochs - 1:
                break
            # Early stopping also exits the while loop (not just the for).
            if (
                es_patience > 0
                and epochs_since_best >= es_patience
                and epoch < epochs - 1
            ):
                break  # break while-loop: early stopping should end training.

    if metrics_logger is not None:
        metrics_logger.close()
    return model

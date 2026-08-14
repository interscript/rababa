"""NaN auto-recovery — detects divergence and resumes from last good state.

When training diverges (val_loss becomes NaN, or gradient norms explode),
this module restores the model + optimizer state from the last known-good
epoch and halves the learning rate, then resumes training.

Usage in train_supervised:

    recovery = NaNAutoRecovery(model, optimizer, scheduler, ckpt_root,
                                max_recoveries=3)
    for epoch in range(start, end):
        ...train one epoch...
        if not math.isnan(val_loss):
            recovery.checkpoint_good(epoch)
        elif recovery.can_recover():
            start = recovery.recover()
            break  # outer loop restarts from new `start`

Why this exists: Hebrew v0.6.0 (zero-centered RMSNorm, pre-fix) silently
diverged around epoch 5-9. The training loop skipped NaN batches but
continued with poisoned momentum, producing a useless checkpoint. This
module gives the loop a way to detect divergence and recover.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class RecoveryStats:
    """Bookkeeping for one recovery attempt."""

    attempt: int
    epoch_recovered_from: int
    lr_before: float
    lr_after: float


class NaNAutoRecovery:
    """Auto-recover from NaN divergence by restoring last good state + halving LR.

    Args:
        model: training model.
        optimizer: training optimizer (any torch.optim.Optimizer or wrapper).
        scheduler: training scheduler (any with step() + state_dict).
        ckpt_root: directory to write recovery logs.
        max_recoveries: max attempts before giving up (default 3).
        lr_scale: factor to scale LR by each recovery (default 0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any | None = None,
        ckpt_root: Path | None = None,
        max_recoveries: int = 3,
        lr_scale: float = 0.5,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ckpt_root = ckpt_root
        self.max_recoveries = max_recoveries
        self.lr_scale = lr_scale
        self._attempts = 0
        self._last_good: dict[str, Any] | None = None
        self.history: list[RecoveryStats] = []

    def checkpoint_good(self, epoch: int, val_loss: float) -> None:
        """Snapshot current state after a successful (non-NaN) epoch.

        Cheap-ish: deep-copies state_dict. For large models this is ~1GB
        of CPU RAM during training — acceptable for our scale (<1B params).
        """
        if math.isnan(val_loss):
            return  # don't snapshot bad states
        self._last_good = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else None,
        }

    def can_recover(self) -> bool:
        """True iff we have a good state to restore AND attempts remaining."""
        return self._last_good is not None and self._attempts < self.max_recoveries

    def recover(self) -> int:
        """Restore last good state, halve LR. Returns epoch to resume from.

        Raises if no good state or max attempts exceeded.
        """
        if self._last_good is None:
            raise RuntimeError("NaNAutoRecovery.recover() called before any good checkpoint")
        if self._attempts >= self.max_recoveries:
            raise RuntimeError(
                f"NaNAutoRecovery exhausted {self.max_recoveries} attempts — giving up"
            )

        # Snapshot current LR before halving.
        lrs_before = [g.get("lr", 0) for g in self._lr_groups()]
        self._attempts += 1

        # Restore state.
        self.model.load_state_dict(self._last_good["model"])
        self.optimizer.load_state_dict(self._last_good["optimizer"])
        if self.scheduler is not None and self._last_good["scheduler"] is not None:
            try:
                self.scheduler.load_state_dict(self._last_good["scheduler"])
            except Exception:
                pass  # scheduler state may not be loadable across versions

        # Halve LR.
        for group in self._lr_groups():
            if "lr" in group:
                group["lr"] *= self.lr_scale
        lrs_after = [g.get("lr", 0) for g in self._lr_groups()]

        stat = RecoveryStats(
            attempt=self._attempts,
            epoch_recovered_from=self._last_good["epoch"],
            lr_before=lrs_before[0] if lrs_before else 0.0,
            lr_after=lrs_after[0] if lrs_after else 0.0,
        )
        self.history.append(stat)

        # Persist recovery log for offline inspection.
        if self.ckpt_root is not None:
            self.ckpt_root.mkdir(parents=True, exist_ok=True)
            log_path = self.ckpt_root / "nan_recovery.log"
            with log_path.open("a", encoding="utf-8") as fh:
                import time
                fh.write(
                    f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] "
                    f"attempt={stat.attempt} "
                    f"restored_epoch={stat.epoch_recovered_from} "
                    f"lr={stat.lr_before:.6f}→{stat.lr_after:.6f}\n"
                )

        return self._last_good["epoch"] + 1

    def _lr_groups(self) -> list[dict[str, Any]]:
        """Return the optimizer's param groups (handles MuonAdamWHybrid too)."""
        if hasattr(self.optimizer, "param_groups"):
            return list(self.optimizer.param_groups)
        # MuonAdamWHybrid wraps Muon + AdamW; expose both.
        groups: list[dict[str, Any]] = []
        for sub in ("muon", "adam"):
            sub_opt = getattr(self.optimizer, sub, None)
            if sub_opt is not None and hasattr(sub_opt, "param_groups"):
                groups.extend(sub_opt.param_groups)
        return groups

    @property
    def attempts(self) -> int:
        return self._attempts

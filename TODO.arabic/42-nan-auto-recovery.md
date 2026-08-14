# 42 — NaN Auto-Recovery (Halve LR + Resume)

## Problem

Hebrew v0.6.0 (zero-centered, before fix) went NaN around epoch 5-9 with no
recovery — the training loop just skipped NaN steps and kept going with
poisoned momentum. The MuonAdamWHybrid kept stepping with degenerate updates
even after individual batches were skipped.

## Fix

Add a `NaNRecovery` wrapper around the optimizer that detects:
1. val_loss = NaN
2. gradient norms exploding (>1e6)
3. weight norms growing >10x init

When triggered:
1. Restore model + optimizer state from last good checkpoint.
2. Halve the learning rate.
3. Resume training from the next epoch.

```python
class NaNAutoRecovery:
    def __init__(self, model, optimizer, scheduler, ckpt_root, lr_scale=0.5):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ckpt_root = ckpt_root
        self.lr_scale = lr_scale
        self._last_good_state = None

    def checkpoint_good(self, epoch: int) -> None:
        """Snapshot current state after a non-NaN epoch."""
        self._last_good_state = {
            "epoch": epoch,
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def recover(self) -> int:
        """Restore from last good state, halve LR. Returns epoch to resume from."""
        if self._last_good_state is None:
            raise RuntimeError("no good state to recover from")
        self.model.load_state_dict(self._last_good_state["model"])
        self.optimizer.load_state_dict(self._last_good_state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] *= self.lr_scale
        return self._last_good_state["epoch"] + 1
```

Wire into `train_supervised`: after each epoch, if val_loss is NaN, call
`recover()` and continue from the returned epoch.

## Files

- `src/rababa/training/recovery.py` (NEW) — `NaNAutoRecovery` class.
- `src/rababa/training/supervised.py:train_supervised` — instantiate + use.
- `tests/training/test_recovery.py` (NEW) — specs.

## Acceptance

- Inject NaN val_loss mid-train → LR halves, resumes from last good epoch.
- Max 3 recovery attempts per training run (then give up + save what we have).
- All existing specs pass.

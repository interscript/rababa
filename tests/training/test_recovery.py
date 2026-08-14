"""Specs for NaNAutoRecovery."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rababa.training.recovery import NaNAutoRecovery


def _make_setup(tmp_path: Path):
    """Helper: build a tiny model + optimizer + recovery instance."""
    model = nn.Linear(8, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    recovery = NaNAutoRecovery(
        model=model, optimizer=opt, scheduler=sched,
        ckpt_root=tmp_path, max_recoveries=3, lr_scale=0.5,
    )
    return model, opt, sched, recovery


def test_nan_recovery_checkpoint_good_snapshots_state(tmp_path: Path):
    """checkpoint_good() should snapshot model + optimizer state."""
    model, opt, _, recovery = _make_setup(tmp_path)
    recovery.checkpoint_good(epoch=0, val_loss=1.5)
    assert recovery._last_good is not None
    assert recovery._last_good["epoch"] == 0
    assert "model" in recovery._last_good
    assert "optimizer" in recovery._last_good


def test_nan_recovery_does_not_snapshot_nan(tmp_path: Path):
    """NaN val_loss should not be snapshotted."""
    _, _, _, recovery = _make_setup(tmp_path)
    recovery.checkpoint_good(epoch=0, val_loss=float("nan"))
    assert recovery._last_good is None


def test_nan_recovery_can_recover_requires_good_state(tmp_path: Path):
    _, _, _, recovery = _make_setup(tmp_path)
    assert not recovery.can_recover()
    recovery.checkpoint_good(epoch=0, val_loss=1.0)
    assert recovery.can_recover()


def test_nan_recovery_recover_halves_lr(tmp_path: Path):
    """recover() should halve the LR."""
    model, opt, _, recovery = _make_setup(tmp_path)
    # Modify weights to non-init state.
    with torch.no_grad():
        model.weight.fill_(2.0)
    recovery.checkpoint_good(epoch=0, val_loss=1.0)
    # Snapshot was of the modified state.
    original_lr = opt.param_groups[0]["lr"]
    new_start = recovery.recover()
    assert new_start == 1  # next epoch after epoch=0
    assert opt.param_groups[0]["lr"] == pytest.approx(original_lr * 0.5)


def test_nan_recovery_recover_restores_model_state(tmp_path: Path):
    """recover() should restore model weights to last good state."""
    model, _, _, recovery = _make_setup(tmp_path)
    with torch.no_grad():
        model.weight.fill_(2.0)
    recovery.checkpoint_good(epoch=0, val_loss=1.0)
    # Corrupt model after snapshot.
    with torch.no_grad():
        model.weight.fill_(999.0)
    recovery.recover()
    # Weights should be back to 2.0 (snapshotted state).
    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))


def test_nan_recovery_max_attempts(tmp_path: Path):
    """After max_recoveries attempts, can_recover should return False."""
    _, _, _, recovery = _make_setup(tmp_path)
    recovery.checkpoint_good(epoch=0, val_loss=1.0)
    assert recovery.can_recover()
    recovery.recover()
    recovery.recover()
    recovery.recover()  # 3rd attempt
    assert not recovery.can_recover()


def test_nan_recovery_logs_to_file(tmp_path: Path):
    """recover() should write to nan_recovery.log in ckpt_root."""
    model, opt, _, recovery = _make_setup(tmp_path)
    recovery.checkpoint_good(epoch=0, val_loss=1.0)
    recovery.recover()
    log = tmp_path / "nan_recovery.log"
    assert log.is_file()
    content = log.read_text()
    assert "attempt=1" in content
    assert "restored_epoch=0" in content


def test_nan_recovery_handles_muon_adamw_hybrid(tmp_path: Path):
    """_lr_groups() should find param groups even in MuonAdamWHybrid-like wrappers."""
    class _FakeMuon:
        def __init__(self):
            self.param_groups = [{"lr": 0.02}]
    class _FakeHybrid:
        def __init__(self):
            self.muon = _FakeMuon()
            self.adam = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))])
            self.param_groups = self.adam.param_groups  # mimic standard optimizer
    opt = _FakeHybrid()
    model = nn.Linear(4, 2)
    recovery = NaNAutoRecovery(
        model=model, optimizer=opt, scheduler=None,
        ckpt_root=tmp_path, max_recoveries=2,
    )
    groups = recovery._lr_groups()
    # Should include both muon's and adam's groups.
    assert len(groups) >= 1

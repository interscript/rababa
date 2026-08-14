"""Specs for multi-seed training launcher."""

from __future__ import annotations

import torch

from rababa.training.multi_seed import _set_seed, teacher_checkpoint_paths


def test_set_seed_makes_torch_deterministic():
    _set_seed(42)
    a = torch.randn(3, 3)
    _set_seed(42)
    b = torch.randn(3, 3)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_produce_different_results():
    _set_seed(42)
    a = torch.randn(3, 3)
    _set_seed(1337)
    b = torch.randn(3, 3)
    assert not torch.equal(a, b)


def test_teacher_checkpoint_paths_returns_existing_files(tmp_path):
    # Create seed-000/run-001/best.pt and seed-001/run-001/best.pt; skip 002.
    task = "rababa_arabic_pro"
    for seed in (0, 1):
        d = tmp_path / task / f"seed-{seed:03d}" / "run-001"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"stub")
    found = teacher_checkpoint_paths(task, n_seeds=3, root=str(tmp_path))
    assert len(found) == 2
    assert all(p.name == "best.pt" for p in found)


def test_teacher_checkpoint_paths_handles_missing_root(tmp_path):
    # Nonexistent root → empty list, no exception.
    found = teacher_checkpoint_paths("no_such_task", n_seeds=3, root=str(tmp_path))
    assert found == []

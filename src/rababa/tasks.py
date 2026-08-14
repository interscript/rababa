"""Task dispatch — single source of truth for cfg → dataset + collate + loader.

Both `modal_app.py` and `cli.py` go through this module so the mapping
from task kind to (dataset, collate) lives in exactly one place. Adding
a new language means adding a branch here, not in every entry point.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .config import load_task_config
from .datasets import (
    load_arabic_mlm,
    load_hebrew_mlm,
    load_nakdimon,
    load_tashkeela,
)
from .training.collate import collate_batch, multi_head_collate_batch
from .training.pretrain import make_mlm_collate_fn


# Map cfg.kind → supervised dataset + collate
SUPERVISED_DATASETS = {
    "rababa": (load_tashkeela, collate_batch),
    "rababa_hebrew": (load_nakdimon, multi_head_collate_batch),
}

# Map cfg.kind → MLM dataset
MLM_DATASETS = {
    "rababa_mlm": load_arabic_mlm,
    "rababa_hebrew_mlm": load_hebrew_mlm,
}


def _get_cleaner(cfg: Any) -> str:
    return cfg.data.get("cleaner", "arabic") if hasattr(cfg.data, "get") else cfg.data.get("cleaner", "arabic")


def _get_max_len(cfg: Any) -> int:
    return int(cfg.model.get("max_len", 200))


def _get_data_root(cfg: Any) -> str | None:
    """Return cfg.data.root if set, else None (let the loader pick its default)."""
    raw = cfg.data.get("root") if hasattr(cfg.data, "get") else None
    return raw or None


def build_supervised_loaders(
    cfg: Any,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) for supervised fine-tune, dispatched by cfg.kind.

    `num_workers` defaults to cfg.train.num_workers or 8 (was 2; bumped for
    ~30-50% training speedup on Modal A100). Also enables persistent_workers
    + pin_memory to keep workers warm + speed up host→GPU transfer.
    """
    kind = cfg.kind
    if kind not in SUPERVISED_DATASETS:
        raise ValueError(
            f"unknown supervised task kind: {kind!r}; "
            f"expected one of {list(SUPERVISED_DATASETS)}"
        )
    loader_fn, collate = SUPERVISED_DATASETS[kind]
    cleaner = _get_cleaner(cfg)
    max_len = _get_max_len(cfg)
    root = _get_data_root(cfg)
    bs = batch_size or int(cfg.train.get("batch_size", 32))
    if num_workers is None:
        num_workers = int(cfg.train.get("num_workers", 8)) if hasattr(cfg.train, "get") else 8

    arch = cfg.get("model", {}).get("arch", "") if hasattr(cfg, "get") else ""

    # Hebrew seq2seq: special data pipeline (undiacritized → diacritized).
    if kind == "rababa_hebrew" and arch == "hebrew_seq2seq":
        from .models.hebrew_seq2seq import (
            HebrewSeq2SeqDataset, hebrew_seq2seq_collate, build_hebrew_vocab,
        )
        from pathlib import Path as _P
        from .datasets import _find_nakdimon_root
        nakdimon_root = root if root else str(_find_nakdimon_root())
        vocab = build_hebrew_vocab(_P(nakdimon_root) / "train.txt")
        # Use data.max_len (200) for sentence filtering, NOT model.max_len (2048)
        # which is only for RoPE cache sizing. Diacritized tgt is ~1.7x src,
        # so 200 undiacritized → ~340 diacritized, well within RoPE=2048.
        data_max_len = int(cfg.data.get("max_len", 200)) if hasattr(cfg.data, "get") else 200
        train_ds = HebrewSeq2SeqDataset(_P(nakdimon_root) / "train.txt", vocab, max_len=data_max_len)
        val_ds = HebrewSeq2SeqDataset(_P(nakdimon_root) / "val.txt", vocab, max_len=data_max_len)
        collate = hebrew_seq2seq_collate
    elif kind == "rababa":
        train_ds = loader_fn("train", root=root, cleaner=cleaner)
        val_ds = loader_fn("val", root=root, cleaner=cleaner)
    elif kind == "rababa_hebrew" and arch == "alephbert":
        from .models.alephbert import AlephBERTHebrewDataset
        train_ds = AlephBERTHebrewDataset("train", root=root, max_len=max_len)
        val_ds = AlephBERTHebrewDataset("val", root=root, max_len=max_len)
    else:  # rababa_hebrew — pass max_len
        train_ds = loader_fn("train", root=root, cleaner=cleaner, max_len=max_len)
        val_ds = loader_fn("val", root=root, cleaner=cleaner, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    return train_loader, val_loader


def build_mlm_loaders(
    cfg: Any,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) for MLM pretrain, dispatched by cfg.kind.

    If `cfg.train.pretrain_method == "mtp"`, uses the MTP collator (pads target
    to T+N-1 so each head can slice `target[:, i:i+T]`).
    """
    kind = cfg.kind
    if kind not in MLM_DATASETS:
        raise ValueError(
            f"unknown MLM task kind: {kind!r}; "
            f"expected one of {list(MLM_DATASETS)}"
        )
    loader_fn = MLM_DATASETS[kind]
    cleaner = _get_cleaner(cfg)
    max_len = _get_max_len(cfg)
    root = _get_data_root(cfg)
    bs = batch_size or int(cfg.train.get("batch_size", 64))
    mask_prob = float(cfg.data.get("mask_prob", 0.15))
    if num_workers is None:
        num_workers = int(cfg.train.get("num_workers", 8)) if hasattr(cfg.train, "get") else 8

    train_ds = loader_fn("train", root=root, cleaner=cleaner, mask_prob=mask_prob, max_len=max_len)
    val_ds = loader_fn("val", root=root, cleaner=cleaner, mask_prob=mask_prob, max_len=max_len)
    method = cfg.train.get("pretrain_method", "mlm") if hasattr(cfg.train, "get") else "mlm"
    if method == "mtp":
        from .training.pretrain_mtp import make_mtp_collate_fn
        n_predict = int(cfg.train.get("mtp_n_predict", 2)) if hasattr(cfg.train, "get") else 2
        collate = make_mtp_collate_fn(max_len, n_predict=n_predict)
    else:
        collate = make_mlm_collate_fn(max_len)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    return train_loader, val_loader


def build_test_loader(
    task: str,
    batch_size: int = 32,
    cleaner: str | None = None,
    max_len: int = 200,
    num_workers: int = 0,
) -> DataLoader:
    """Build a test DataLoader for evaluation/benchmarking."""
    cfg = load_task_config(task)
    kind = cfg.kind
    if kind not in SUPERVISED_DATASETS:
        raise ValueError(f"unknown task kind: {kind!r}")
    loader_fn, collate = SUPERVISED_DATASETS[kind]
    cleaner = cleaner or _get_cleaner(cfg)
    root = _get_data_root(cfg)

    if kind == "rababa":
        ds = loader_fn("test", root=root, cleaner=cleaner)
    else:
        ds = loader_fn("test", root=root, cleaner=cleaner, max_len=max_len)

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate,
    )

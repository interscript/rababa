"""Collation — pad sequences to batch max, build length tensor.

Used by DataLoader with `collate_fn=collate_batch` (single-head Arabic)
or `collate_fn=multi_head_collate_batch` (multi-head Hebrew). Both
produce the same `Batch` shape so the training loop is identical
downstream.

Truncates input/target to `max_len` if specified (default: model's
trained max_len, e.g. 200 for rababa_arabic). This matches what
production inference does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import PAD_ID
from ..datasets import Example, HebrewExample


@dataclass
class Batch:
    """Collated batch — works for single-head and multi-head tasks.

    `targets` is always a list. Single-head tasks have one entry;
    multi-head tasks have N (one per output head, in the same order
    as `model.head_names()`).
    """
    src: torch.Tensor              # (batch, seq) int64
    lengths: torch.Tensor          # (batch,) int64
    targets: list[torch.Tensor]    # list of (batch, seq) int64 — one per head
    raw: list[str]


def _truncate_single(batch: list[Example], max_len: int) -> list[Example]:
    out: list[Example] = []
    for ex in batch:
        if len(ex.input_ids) > max_len:
            out.append(Example(
                input_ids=ex.input_ids[:max_len],
                target_ids=ex.target_ids[:max_len],
                raw=ex.raw,
            ))
        else:
            out.append(ex)
    return out


def _truncate_multi(batch: list[HebrewExample], max_len: int) -> list[HebrewExample]:
    out: list[HebrewExample] = []
    for ex in batch:
        if len(ex.input_ids) > max_len:
            out.append(HebrewExample(
                input_ids=ex.input_ids[:max_len],
                niqqud_ids=ex.niqqud_ids[:max_len],
                dagesh_ids=ex.dagesh_ids[:max_len],
                sin_ids=ex.sin_ids[:max_len],
                raw=ex.raw,
            ))
        else:
            out.append(ex)
    return out


def collate_batch(batch: list[Example], max_len: int = 200) -> Batch:
    """Pad sequences to the max length in the batch (after truncating each).

    Single-head: targets has 1 entry.
    """
    batch = _truncate_single(batch, max_len)
    max_actual = max(len(ex.input_ids) for ex in batch)
    src = torch.full((len(batch), max_actual), PAD_ID, dtype=torch.long)
    target = torch.zeros((len(batch), max_actual), dtype=torch.long)
    lengths = torch.zeros((len(batch),), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        src[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        target[i, :n] = torch.tensor(ex.target_ids, dtype=torch.long)
        lengths[i] = n
    return Batch(src=src, lengths=lengths, targets=[target], raw=[ex.raw for ex in batch])


def multi_head_collate_batch(batch: list[HebrewExample], max_len: int = 200) -> Batch:
    """Pad Hebrew multi-head sequences. targets has 3 entries: [niqqud, dagesh, sin]."""
    batch = _truncate_multi(batch, max_len)
    max_actual = max(len(ex.input_ids) for ex in batch)
    bsz = len(batch)
    src = torch.full((bsz, max_actual), PAD_ID, dtype=torch.long)
    niqqud = torch.zeros((bsz, max_actual), dtype=torch.long)
    dagesh = torch.zeros((bsz, max_actual), dtype=torch.long)
    sin = torch.zeros((bsz, max_actual), dtype=torch.long)
    lengths = torch.zeros((bsz,), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        src[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        niqqud[i, :n] = torch.tensor(ex.niqqud_ids, dtype=torch.long)
        dagesh[i, :n] = torch.tensor(ex.dagesh_ids, dtype=torch.long)
        sin[i, :n] = torch.tensor(ex.sin_ids, dtype=torch.long)
        lengths[i] = n
    return Batch(
        src=src, lengths=lengths,
        targets=[niqqud, dagesh, sin],
        raw=[ex.raw for ex in batch],
    )


def make_collate_fn(max_len: int = 200, multi_head: bool = False):
    """Return a collate_fn with a bound max_len."""
    fn = multi_head_collate_batch if multi_head else collate_batch
    def _collate(batch) -> Batch:
        return fn(batch, max_len=max_len)
    return _collate

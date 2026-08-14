"""Noisy Student self-training.

After supervised training, run the model on unlabeled text. Keep the
predictions where the model is high-confidence (per-token softmax max
> threshold). Treat those as silver labels. Add to the training set.
Retrain.

The "noisy" part: augment the SILVER side with input-side noise (char
dropout, keyboard confusables). The model learns to be invariant to
the noise.

Pipeline:
  1. Teacher (the current trained model) labels unlabeled text.
  2. Confidence filter keeps only high-confidence silver.
  3. Augment: apply noise to silver-side inputs.
  4. Combine with original gold training set.
  5. Train a fresh student on combined set.
  6. Student becomes teacher for next round.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..constants import PAD_ID
from .augment import AugmentPipeline, CharDropout, KeyboardConfusables
from .collate import Batch
from .supervised import train_supervised


@dataclass
class SilverExample:
    """A self-labeled example with confidence score."""
    input_ids: list[int]
    target_ids: list[int]
    confidence: float   # mean per-token softmax max
    raw: str


@torch.no_grad()
def label_unlabeled(
    teacher: nn.Module,
    text_lines: Iterable[str],
    encoder,
    batch_size: int = 32,
    max_len: int = 200,
    conf_threshold: float = 0.95,
    device: torch.device | None = None,
) -> list[SilverExample]:
    """Run teacher inference on unlabeled text; return high-confidence silver.

    Args:
        teacher: trained diacritization model (eval mode is set inside).
        text_lines: iterable of raw input strings.
        encoder: a rababa.encoder.ArabicEncoder / HebrewEncoder for input prep.
        batch_size: inference batch size.
        max_len: max sequence length.
        conf_threshold: keep predictions only where mean per-token softmax
            max exceeds this. 0.95 = very conservative.
        device: torch device.

    Returns:
        List of SilverExample with confidence > threshold.
    """
    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval()
    out: list[SilverExample] = []

    # Buffer batches.
    buf_raw: list[str] = []
    buf_ids: list[list[int]] = []

    def _flush() -> None:
        if not buf_ids:
            return
        B = len(buf_ids)
        T = max(len(x) for x in buf_ids)
        src = torch.full((B, T), PAD_ID, dtype=torch.long, device=device)
        lengths = torch.zeros((B,), dtype=torch.long, device=device)
        for i, ids in enumerate(buf_ids):
            src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            lengths[i] = len(ids)
        # Teacher forward: single-head models return list of one tensor.
        outputs = teacher.forward_heads(src, lengths)
        logits = outputs[0]  # primary diacritization head
        probs = torch.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        for i in range(B):
            non_pad = src[i] != PAD_ID
            mean_conf = conf[i][non_pad].mean().item()
            if mean_conf >= conf_threshold:
                pred_ids = preds[i][: int(lengths[i])].tolist()
                out.append(SilverExample(
                    input_ids=buf_ids[i],
                    target_ids=pred_ids,
                    confidence=mean_conf,
                    raw=buf_raw[i],
                ))
        buf_raw.clear()
        buf_ids.clear()

    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        cleaned = encoder.clean(line)
        if not cleaned:
            continue
        ids = encoder.encode(cleaned)[:max_len]
        if len(ids) < 4:
            continue
        buf_raw.append(line)
        buf_ids.append(ids)
        if len(buf_ids) >= batch_size:
            _flush()
    _flush()
    return out


class CombinedDataset(Dataset):
    """Concatenates gold supervised examples with silver self-labeled ones.

    Both must conform to the same Example shape (input_ids, target_ids, raw).
    The silver side is optionally augmented via `augment` per __getitem__.
    """

    def __init__(
        self,
        gold_examples: list,            # list of Example (rababa.datasets.Example)
        silver_examples: list[SilverExample],
        augment: AugmentPipeline | None = None,
        silver_upweight: int = 1,
    ) -> None:
        # Convert silver to a dict-shaped example matching gold's interface.
        from ..datasets import Example
        self.gold: list[Example] = list(gold_examples)
        self.silver: list[Example] = [
            Example(input_ids=s.input_ids, target_ids=s.target_ids, raw=s.raw)
            for s in silver_examples for _ in range(silver_upweight)
        ]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.gold) + len(self.silver)

    def __getitem__(self, idx: int):
        if idx < len(self.gold):
            ex = self.gold[idx]
        else:
            ex = self.silver[idx - len(self.gold)]
        if self.augment is not None:
            aug_ids = self.augment(ex.input_ids)
            # Keep length matching target length.
            n = min(len(aug_ids), len(ex.target_ids))
            from ..datasets import Example
            return Example(input_ids=aug_ids[:n], target_ids=ex.target_ids[:n], raw=ex.raw)
        return ex


def noisy_student_round(
    task: str,
    teacher_checkpoint: Path,
    unlabeled_path: Path,
    ckpt_root: Path,
    cfg: dict[str, Any],
    device: torch.device,
    conf_threshold: float = 0.95,
    augment: AugmentPipeline | None = None,
    log_fn=None,
) -> Path:
    """Run one round of noisy-student self-training.

    Returns: path to the new student's best.pt.
    """
    from ..config import load_task_config, to_dict
    from ..models.base import build_model
    from ..tasks import SUPERVISED_DATASETS

    full_cfg = load_task_config(task)
    cfg_dict = to_dict(full_cfg)
    kind = full_cfg.kind
    if kind not in SUPERVISED_DATASETS:
        raise ValueError(f"unknown task kind: {kind!r}")
    loader_fn, _ = SUPERVISED_DATASETS[kind]
    cleaner = "hebrew" if "hebrew" in task else "arabic"
    root = full_cfg.data.get("root") if hasattr(full_cfg.data, "get") else None
    if kind == "rababa":
        gold_train = loader_fn("train", root=root, cleaner=cleaner)
    else:
        gold_train = loader_fn("train", root=root, cleaner=cleaner, max_len=200)

    # Load teacher.
    teacher = build_model(cfg_dict).to(device)
    state = torch.load(teacher_checkpoint, map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    teacher.load_state_dict(state)

    # Encoder for unlabeled text.
    if "hebrew" in task:
        from ..encoder import HebrewEncoder
        enc = HebrewEncoder(cleaner="hebrew")
    else:
        from ..encoder import ArabicEncoder
        enc = ArabicEncoder(cleaner="arabic")

    # Label unlabeled text.
    with unlabeled_path.open(encoding="utf-8") as f:
        text_lines = (ln for ln in f if ln.strip())
        silver = label_unlabeled(
            teacher, text_lines, enc,
            batch_size=32, max_len=200,
            conf_threshold=conf_threshold, device=device,
        )
    print(f"[noisy-student] labeled {len(silver):,} high-confidence silver examples")

    # Combine into a single dataset.
    combined = CombinedDataset(gold_train.examples, silver, augment=augment)

    # Build new train/val loaders from the combined dataset.
    from .collate import collate_batch, multi_head_collate_batch
    collate = multi_head_collate_batch if kind == "rababa_hebrew" else collate_batch
    val_ds = loader_fn("val", root=root, cleaner=cleaner) if kind == "rababa" else loader_fn("val", root=root, cleaner=cleaner, max_len=200)
    train_loader = DataLoader(combined, batch_size=int(cfg_dict["train"].get("batch_size", 32)),
                              shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(cfg_dict["train"].get("batch_size", 32)),
                            shuffle=False, num_workers=2, collate_fn=collate)

    # Train fresh student.
    student = train_supervised(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg_dict,
        device=device,
        ckpt_root=ckpt_root,
        log_fn=log_fn,
    )

    best_path = ckpt_root / "best.pt"
    return best_path

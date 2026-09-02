"""Per-epoch metrics logger — JSONL on volume, structured for offline analysis.

Sister to VolumeLogger (which writes human-readable log lines). MetricsLogger
writes one JSON object per epoch, enabling:

  - Plotting loss curves from outside Modal.
  - Detecting divergence early (NaN val_loss, sudden spikes).
  - Comparing runs (multi-seed, hyperparameter sweeps).

File format (one JSON object per line, JSONL):

    {"epoch": 0, "train_loss": 4.21, "val_loss": 4.55, "learning_rate": 0.0003, "ts": 1234567890}
    {"epoch": 1, "train_loss": 3.85, "val_loss": 4.12, "learning_rate": 0.00028, "ts": 1234567990}
    ...

Writes to /tmp first then syncs to the volume (same pattern as VolumeLogger)
to avoid blocking the Modal volume's reload during the training loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EpochMetrics:
    """One row of metrics.jsonl."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    ts: float

    @classmethod
    def from_train_metrics(cls, m: Any) -> "EpochMetrics":
        """Build from a TrainMetrics dataclass (rababa.training.supervised).

        TrainMetrics has fields: epoch, train_loss, val_loss, learning_rate.
        """
        return cls(
            epoch=int(m.epoch),
            train_loss=float(m.train_loss),
            val_loss=float(m.val_loss),
            learning_rate=float(m.learning_rate),
            ts=time.time(),
        )

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


class MetricsLogger:
    """Append-only JSONL logger for per-epoch metrics.

    Usage:
        logger = MetricsLogger(volume_root / "metrics.jsonl")
        for epoch in range(N):
            ...
            logger.log(TrainMetrics(epoch=epoch, train_loss=..., val_loss=..., lr=...))
        logger.close()  # final sync to volume

    Reads are easy:
        Path("metrics.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in lines]
    """

    def __init__(self, volume_path: Path) -> None:
        self.volume_path = volume_path
        # Mirror to /tmp for the same reason as VolumeLogger: avoid blocking
        # volume.reload() during training. In tests (where volume_path is
        # already in a tmp dir), use the path directly.
        if str(volume_path).startswith("/tmp") or volume_path.parent.is_dir():
            self.local_path = volume_path
        else:
            self.local_path = Path("/tmp") / volume_path.name
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_path.is_file():
            self.local_path.touch()

    def log(self, metrics: Any) -> None:
        """Append one metrics row. Accepts TrainMetrics or EpochMetrics."""
        if isinstance(metrics, EpochMetrics):
            row = metrics
        else:
            row = EpochMetrics.from_train_metrics(metrics)
        with self.local_path.open("a", encoding="utf-8") as fh:
            fh.write(row.to_json_line() + "\n")
        # Best-effort sync after each epoch (epochs are infrequent).
        self.sync_to_volume()

    def sync_to_volume(self) -> None:
        self.volume_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.volume_path.write_text(
                self.local_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except OSError:
            pass

    def close(self) -> None:
        self.sync_to_volume()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all rows. Useful for in-pipeline inspection / benchmarks."""
        if not self.local_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.local_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

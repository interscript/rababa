"""Specs for MetricsLogger (per-epoch JSONL metrics)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rababa.training.metrics import EpochMetrics, MetricsLogger


@dataclass
class _MockTrainMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float


def test_epoch_metrics_serializes_to_json():
    m = EpochMetrics(epoch=0, train_loss=4.2, val_loss=4.5, learning_rate=0.0003, ts=12345.0)
    line = m.to_json_line()
    parsed = json.loads(line)
    assert parsed["epoch"] == 0
    assert parsed["train_loss"] == 4.2
    assert parsed["val_loss"] == 4.5
    assert parsed["learning_rate"] == 0.0003
    assert parsed["ts"] == 12345.0


def test_epoch_metrics_from_train_metrics():
    m = _MockTrainMetrics(epoch=1, train_loss=3.0, val_loss=3.2, learning_rate=0.00025)
    em = EpochMetrics.from_train_metrics(m)
    assert em.epoch == 1
    assert em.train_loss == 3.0
    assert em.val_loss == 3.2
    assert em.learning_rate == 0.00025
    assert em.ts > 0


def test_metrics_logger_writes_jsonl(tmp_path: Path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path)
    logger.log(_MockTrainMetrics(epoch=0, train_loss=4.0, val_loss=4.5, learning_rate=0.001))
    logger.log(_MockTrainMetrics(epoch=1, train_loss=3.5, val_loss=4.0, learning_rate=0.0009))
    logger.close()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["epoch"] == 0
    assert rows[1]["epoch"] == 1
    assert rows[1]["train_loss"] < rows[0]["train_loss"]


def test_metrics_logger_read_all_roundtrip(tmp_path: Path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path)
    for ep in range(5):
        logger.log(_MockTrainMetrics(epoch=ep, train_loss=4.0 - ep * 0.5,
                                      val_loss=4.5 - ep * 0.4, learning_rate=0.001))
    logger.close()
    rows = logger.read_all()
    assert len(rows) == 5
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3, 4]
    # Verify val_loss is decreasing (typical training pattern).
    assert rows[-1]["val_loss"] < rows[0]["val_loss"]


def test_metrics_logger_appends_to_existing(tmp_path: Path):
    """If the file already has rows, new ones should append (not truncate)."""
    log_path = tmp_path / "metrics.jsonl"
    log_path.write_text(json.dumps({"epoch": 0, "train_loss": 5.0, "val_loss": 5.5,
                                     "learning_rate": 0.001, "ts": 1.0}) + "\n")
    logger = MetricsLogger(log_path)
    logger.log(_MockTrainMetrics(epoch=1, train_loss=4.0, val_loss=4.5, learning_rate=0.0009))
    logger.close()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_metrics_logger_no_path_does_not_crash():
    """If metrics_path=None, the training loop should still run without metrics logging.

    The caller (training loop) checks `if metrics_logger is not None`,
    so this is implicit. The MetricsLogger itself always requires a path.
    This test documents that contract.
    """
    with pytest.raises((TypeError, AttributeError)):
        MetricsLogger(None)  # type: ignore[arg-type]


def test_metrics_logger_detects_divergence(tmp_path: Path):
    """Use case: detect NaN val_loss mid-training (the Hebrew v0.6.0 incident).

    A monitoring script can read the metrics file and raise an alert if
    val_loss becomes NaN. This spec documents that NaN is preserved in
    the JSONL output.
    """
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path)
    logger.log(_MockTrainMetrics(epoch=0, train_loss=4.0, val_loss=4.5, learning_rate=0.001))
    # Epoch 5: NaN — model diverged.
    logger.log(_MockTrainMetrics(epoch=5, train_loss=float("nan"), val_loss=float("nan"),
                                  learning_rate=0.0005))
    logger.close()
    rows = logger.read_all()
    assert len(rows) == 2
    assert rows[0]["val_loss"] == 4.5
    # JSON serializes NaN as null/NaN — caller should detect this.
    nan_val = rows[1]["val_loss"]
    assert nan_val is None or str(nan_val).lower() == "nan"

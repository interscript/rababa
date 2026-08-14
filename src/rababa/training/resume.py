"""Resume + status helpers for resilient Modal training.

Three concerns when the local session disconnects mid-sprint:

1. **Mid-training resume**: a Modal function that dies at epoch 5/10
   should not restart from epoch 0 on retry. `latest_resume_checkpoint`
   finds the highest-epoch checkpoint in a directory; training loops
   call this on startup to continue where they left off.

2. **Stage status tracking**: when `train_all.py` re-runs, it should
   skip stages that already completed on the volume. `mark_stage_done`
   and `is_stage_done` write/read a JSON index on the checkpoints
   volume root (`/checkpoints/_status.json`).

3. **Log persistence**: every training run writes a log file on the
   checkpoints volume alongside the per-epoch .pt files, so even if
   the local run.log is lost, the volume has a copy.

All helpers are no-ops if the volume isn't mounted (e.g., in local
dev). They never raise on missing files — they return None / False.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

EPOCH_CKPT_RE = re.compile(r"checkpoint-epoch-(\d+)\.pt$")


def latest_resume_checkpoint(ckpt_root: Path) -> tuple[Path, int] | None:
    """Find the highest-epoch checkpoint in `ckpt_root`.

    Returns (path, epoch) or None if no checkpoints exist.
    Looks for `checkpoint-epoch-N.pt` files; falls back to `best.pt`
    if only that exists (epoch inferred as -1, meaning "resume unknown").
    """
    if not ckpt_root.is_dir():
        return None
    epoch_ckpts = []
    for p in ckpt_root.glob("checkpoint-epoch-*.pt"):
        m = EPOCH_CKPT_RE.search(p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))
    if epoch_ckpts:
        epoch_ckpts.sort()
        return epoch_ckpts[-1][1], epoch_ckpts[-1][0]
    best = ckpt_root / "best.pt"
    if best.is_file():
        return best, -1
    return None


def load_resume_state(
    model,
    optimizer,
    scheduler,
    path: Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore model + optimizer + scheduler + epoch.

    Returns the checkpoint dict (contains 'epoch', 'best_val_loss', etc.).
    Optimizer and scheduler are optional — pass None to skip restoring them.

    Tolerates two checkpoint formats:
      - Resumable dict: `{"model": ..., "optimizer": ..., ...}` (current)
      - Raw state_dict: `{param_name: tensor, ...}` (legacy best.pt saves)
    The legacy format only restores model weights — no optimizer/scheduler.
    """
    state = torch.load(path, map_location=device, weights_only=False) if device else torch.load(path, weights_only=False)
    # Detect legacy raw state_dict (no "model" key, has parameter-tensor values).
    if "model" in state:
        model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
        return state
    # Legacy: state IS the state_dict.
    model.load_state_dict(state)
    return {"model": state, "epoch": -1, "best_val_loss": None}


def save_resumable_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint with full resume state. Inverse of load_resume_state."""
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ---- Stage status index ----------------------------------------------


def _status_path(volume_root: Path) -> Path:
    return volume_root / "_status.json"


def read_status(volume_root: Path) -> dict[str, Any]:
    """Read the stage-status index. Returns {} if missing or unreadable."""
    p = _status_path(volume_root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_stage_done(volume_root: Path, stage: str) -> bool:
    """True iff `stage` was previously marked done in the status index."""
    return bool(read_status(volume_root).get("stages", {}).get(stage, {}).get("done"))


def mark_stage_done(
    volume_root: Path,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append/overwrite `stage` in the status index as done.

    Safe to call repeatedly — overwrites prior entry with new timestamp.
    """
    status = read_status(volume_root)
    status.setdefault("stages", {})
    entry: dict[str, Any] = {"done": True, "ts": time.time()}
    if extra:
        entry.update(extra)
    status["stages"][stage] = entry
    _status_path(volume_root).parent.mkdir(parents=True, exist_ok=True)
    _status_path(volume_root).write_text(json.dumps(status, indent=2), encoding="utf-8")


def mark_stage_failed(
    volume_root: Path,
    stage: str,
    error: str,
) -> None:
    """Mark `stage` as failed with error message. Does not set done=True."""
    status = read_status(volume_root)
    status.setdefault("stages", {})
    status["stages"][stage] = {
        "done": False,
        "ts": time.time(),
        "error": error[:500],  # truncate to keep JSON manageable
    }
    _status_path(volume_root).parent.mkdir(parents=True, exist_ok=True)
    _status_path(volume_root).write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---- Volume log file --------------------------------------------------


class VolumeLogger:
    """Tee writes to stdout + a local file.

    Writes go to `/tmp/...` (container-local, NOT on a volume) so that
    Modal's `volume.reload()` works without "open files" errors. The
    `sync_to_volume(dst_path)` method copies the local log to a volume
    path at safe points (between stages, never during a reload).

    Used inside Modal functions so that even if the local session is
    disconnected, the log file persists on the volume for later retrieval.
    """

    def __init__(self, log_path: Path) -> None:
        """Args:
            log_path: INTENDED volume path (used by sync_to_volume and as
                the canonical location). Day-to-day writes go to a /tmp
                mirror to avoid blocking volume.reload().
        """
        self.volume_path = log_path
        # Mirror path: same filename under /tmp.
        self.local_path = Path("/tmp") / log_path.name
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_path.is_file():
            self.local_path.touch()

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.local_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def sync_to_volume(self) -> None:
        """Copy local log to the volume path. Call between stages."""
        self.volume_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.volume_path.write_text(self.local_path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass  # best-effort

    def close(self) -> None:
        self.sync_to_volume()


# Late import to keep this module dependency-light for non-PyTorch callers.
import torch  # noqa: E402  (intentional late import)

# Re-export MetricsLogger so callers can do `from .resume import VolumeLogger, MetricsLogger`.
from .metrics import MetricsLogger  # noqa: E402  (intentional late import)

"""Freeze the private Arabic dev set (TODO.research/12: honest selection).

Public benchmarks are measured ONCE at the end of a recipe; day-to-day
model selection uses this frozen split so we stop picking recipes by
dev-set noise (the Persian v1-beats-v3/v4/v5 lesson).

The dev set is the exact held-out val slice of the ByT5 r2 run (seed-42
shuffle of arabic-combined/train.txt, first 2,000 lines, ≤640-byte
filter) — derivation is deterministic, the manifest makes it tamper-
evident. RAFT and future fine-tunes select on this; SadeedDiac-25 stays
untouched until the final number.

Usage:
    modal run scripts/make_private_dev.py
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

N_VAL = 2_000
MAX_BYTES = 640
CORPUS = "/datasets/arabic-combined/train.txt"
OUT_DIR = Path("/datasets/private-dev/arabic")

image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("rababa-private-dev", image=image)


def _pairs() -> list[tuple[str, str]]:
    diacritics = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
    lines = [
        l.strip()
        for l in Path(CORPUS).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    random.Random(42).shuffle(lines)
    pairs: list[tuple[str, str]] = []
    for line in lines[:N_VAL]:
        src = diacritics.sub("", line)
        if not src:
            continue
        if len(src.encode("utf-8")) > MAX_BYTES or len(line.encode("utf-8")) > MAX_BYTES:
            continue
        pairs.append((src, line))
    return pairs


@app.function(volumes={"/datasets": datasets_volume}, timeout=30 * 60)
def freeze_arabic_dev() -> dict:
    import json

    if (OUT_DIR / "FROZEN").exists():
        return {"status": "already-frozen", "dev": str(OUT_DIR / "dev.jsonl")}

    pairs = _pairs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_path = OUT_DIR / "dev.jsonl"
    dev_path.write_text(
        "".join(json.dumps({"src": s, "gold": g}, ensure_ascii=False) + "\n" for s, g in pairs),
        encoding="utf-8",
    )
    digest = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    manifest = OUT_DIR / "MANIFEST.txt"
    manifest.write_text(
        f"dataset: arabic private dev\n"
        f"derived_from: {CORPUS} (seed-42 shuffle, first {N_VAL} lines, <= {MAX_BYTES} bytes)\n"
        f"lines: {len(pairs)}\n"
        f"sha256: {digest}\n"
        f"frozen: deterministic derivation — do not regenerate with different params\n",
        encoding="utf-8",
    )
    (OUT_DIR / "FROZEN").touch()
    datasets_volume.commit()
    return {"status": "frozen", "lines": len(pairs), "sha256": digest}

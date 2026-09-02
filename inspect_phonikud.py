"""Inspect phonikud knesset_nikud_v6 format: column layout, marker inventory,
length stats. Read-only; no writes.

Usage:
    modal run inspect_phonikud.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
SRC = Path("/datasets/hebrew-phonikud/knesset_nikud_v6.txt")

image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("rababa-inspect-phonikud", image=image)

NIKUD = set("ׇ֑") | {chr(c) for c in range(0x05B0, 0x05C0)}


@app.function(timeout=20 * 60, volumes={"/datasets": datasets_volume})
def inspect() -> dict:
    datasets_volume.reload()

    with open(SRC, encoding="utf-8") as f:
        head = [next(f) for _ in range(5)]
    for i, line in enumerate(head):
        cols = line.rstrip("\n").split("\t")
        print(f"[line {i}] n_cols={len(cols)}", flush=True)
        for j, c in enumerate(cols):
            print(f"  col{j}: {c[:120]!r}", flush=True)

    tabs = Counter()
    col2_markers = Counter()
    col2_has_nikud = col1_has_nikud = 0
    col1_lens = []
    n = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            n += 1
            if n > 200_000:
                break
            cols = line.rstrip("\n").split("\t")
            tabs[len(cols)] += 1
            if len(cols) != 2:
                continue
            t, p = cols
            col1_lens.append(len(t))
            if any(ch in NIKUD for ch in t):
                col1_has_nikud += 1
            if any(ch in NIKUD for ch in p):
                col2_has_nikud += 1
            for ch in p:
                if not ("א" <= ch <= "ת") and ch != " ":
                    col2_markers[ch] += 1

    col1_lens.sort()
    print(f"[stats] n={n} tabs_per_line={dict(tabs)}", flush=True)
    print(f"[stats] col1_nikud={col1_has_nikud} col2_nikud={col2_has_nikud}", flush=True)
    if col1_lens:
        print(f"[stats] col1_len p50={col1_lens[len(col1_lens)//2]} "
              f"p95={col1_lens[int(len(col1_lens)*0.95)]} max={col1_lens[-1]}", flush=True)
    print(f"[stats] col2 non-letter markers: {col2_markers.most_common(20)}", flush=True)
    return {"n": n, "tabs": dict(tabs)}


@app.local_entrypoint()
def main():
    print(inspect.remote())

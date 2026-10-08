# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "onnxruntime==1.20.1",
#   "numpy<2",
# ]
# ///
"""WO26 stage 1: heb-diac-plane-2.0 pseudo-labels hewiki (Hebrew
noisy student). Teacher = our shipped 8.18-DER artifact (NO LLM).
Windows of 1400 bytes, greedy K-pass via the CPU ORT runtime;
every unit written with a margin proxy = fraction of chars whose
top-1/top-2 logit gap exceeds the threshold (kept at train time).

Usage:
    hf jobs uv run --flavor l4x1 -d --timeout 12h \\
      -v hf://datasets/Interscript/hebrew-v4:/train_data:ro \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -v /tmp/isx-hf-code:/code:ro label_hewiki_hf.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/code")
sys.path.insert(0, "/code/src")

DATA = Path("/train_data")
CKPT = Path("/ckpt/run-030-heb-noisy")
ZIP = Path("/ckpt/run-022-heb-plane-base/export/heb-diac-plane-2.0.zip")
UNIT_BYTES = 1400
MARGIN_NAT = 1.0
MAX_UNITS = 40_000

import numpy as np
import onnxruntime as ort

import nikud_planes as NP
from interscript.ml.plane import PlaneModel


def window(text: str, budget: int = UNIT_BYTES) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words, cur, n, wins = text.split(), [], 0, []
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    model = PlaneModel.from_zip(ZIP.read_bytes())

    units: list[str] = []
    for name in ("train.txt", "val.txt"):
        p = DATA / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            skel = NP.split_planes(line)[0]
            if len(skel) < 10:
                continue
            units.extend(window(skel))
            if len(units) >= MAX_UNITS:
                break
        if len(units) >= MAX_UNITS:
            break
    print(f"[data] hewiki skeleton units={len(units)}", flush=True)

    # margin: rerun logits once more than K passes and read the gap
    import interscript.ml.plane as PL

    out_path = CKPT / "hewiki-pseudo.jsonl"
    sess = model.sess
    classes = model.classes
    n_classes = model.n_classes
    done = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for text in units:
            skel = text
            ids = [b + 3 for b in skel.encode("utf-8")]
            t_ids = np.array([ids], dtype=np.int64)
            plane = np.full_like(t_ids, n_classes)
            margins = None
            for p_i in range(model.k):
                logits = sess.run(None, {"input_ids": t_ids, "plane_ids": plane})[0]
                flat = logits[0]
                top2 = np.partition(flat, -2, axis=-1)[..., -2:]
                gap = top2[..., 1] - top2[..., 0]
                margins = gap if margins is None else np.minimum(margins, gap)
                plane = logits.argmax(-1)
            # fraction of byte positions above the margin threshold
            frac = float((margins > MARGIN_NAT).mean())
            pos = 0
            char_preds = []
            for ch in skel:
                n = len(ch.encode("utf-8"))
                votes = plane[0, pos:pos + n].tolist()
                cid = max(set(votes), key=votes.count)
                char_preds.append(classes[cid] if cid < n_classes else "")
                pos += n
            labeled = NP.render(skel, char_preds)
            done += 1
            out.write(json.dumps({"src": skel, "tgt": labeled,
                                  "keep_frac": round(frac, 4)},
                                 ensure_ascii=False) + "\n")
            if done % 200 == 0:
                out.flush()
                print(f"[gen] {done}/{len(units)} "
                      f"({done / (time.time() - t0):.1f} win/s)", flush=True)
    print(f"[done] {done} windows -> {out_path}", flush=True)
    (CKPT / "LABEL_DONE").write_text(f"total={done}\n")


if __name__ == "__main__":
    main()

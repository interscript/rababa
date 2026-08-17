"""SadeedDiac-25 eval with windowed generation (no truncation losses).

The r2/r3 evals capped generation at 1024 bytes: 57/1,200 benchmark
paragraphs were hard-truncated (missing tails scored as errors) and 277
exceed ByT5's 640-byte training window. Here, inputs longer than 600
bytes are split at word boundaries into in-distribution windows,
generated greedily per window, and stitched. Short inputs stay
single-shot.

Usage:
    modal run --detach eval_sadeed_windowed.py
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

MODEL_DIR = "/checkpoints/rababa_arabic_byt5/run-003-domain/best"
TAG = "r3_windowed"
WINDOW = 600

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "pyarabic", "prettytable", "pandas", "pyarrow")
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-windowed-eval", image=image)


def split_windows(text: str, budget: int = WINDOW) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words = text.split()
    wins: list[str] = []
    cur: list[str] = []
    n = 0
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur))
            cur, n = [], 0
        cur.append(w)
        n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/checkpoints": checkpoints_volume})
def evaluate() -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoints_volume.reload()
    out_dir = Path("/checkpoints/rababa_arabic_byt5/run-003-domain")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to("cuda")
    model.eval()
    device = next(model.parameters()).device

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    all_windows: list[str] = []
    counts: list[int] = []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    n_win = sum(1 for c in counts if c > 1)
    print(f"[data] {len(inputs)} paragraphs, {len(all_windows)} windows ({n_win} multi-window)", flush=True)

    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(all_windows), 32):
            batch = all_windows[i : i + 32]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=WINDOW
            ).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=WINDOW, num_beams=1)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            if (i // 32) % 20 == 0:
                print(f"[gen] {i + len(batch)}/{len(all_windows)}", flush=True)

    k = 0
    paragraphs: list[str] = []
    for c in counts:
        paragraphs.append(" ".join(preds[k : k + c]))
        k += c

    csv_path = Path(f"/tmp/sadeed_{TAG}.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (out_dir / f"sadeed_preds_{TAG}.csv").write_text(csv_path.read_text(), encoding="utf-8")
    checkpoints_volume.commit()

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"\n===== {TAG} (their default protocol) =====", flush=True)
    E.report_errors_on_csv_file(
        str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
        gt_missing_diacritic_is_error=False,
    )
    checkpoints_volume.commit()
    return {"tag": TAG}


@app.local_entrypoint()
def main():
    evaluate.remote()

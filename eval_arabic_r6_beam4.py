"""Arabic r6 beam-4 probe (TODO.public §4).

All Arabic numbers are greedy; Hebrew gained 12 DER points from beam.
Rerun r6's exact windowed zero-skip SadeedDiac-25 eval with beam-4
from the frozen run-006-morph best. If DER drops, ship beam — free
quality, zero training. Resumable per-window.

Usage:
    modal run --detach eval_arabic_r6_beam4.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_byt5/run-006-morph"
UNIT_BYTES = 1400
GREEDY_DER = 2.5793

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "pandas",
        "pyarrow",
        "pyarabic",
        "prettytable",
    )
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-r6-beam4", image=image)


@app.function(
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def evaluate() -> dict:
    import torch
    import pandas as pd
    import pyarrow.parquet as pq
    from difflib import SequenceMatcher
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer

    checkpoints_volume.reload()
    out_dir = Path("/checkpoints") / RUN
    done_marker = out_dir / "BEAM4_DONE"
    if done_marker.exists():
        fe = out_dir / "beam4_eval.json"
        return json.loads(fe.read_text(encoding="utf-8")) if fe.exists() else {"status": "already-done"}

    ckpt = str(out_dir / "best")
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to("cuda")
    tokenizer = ByT5Tokenizer.from_pretrained(ckpt)
    model.eval()

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    def split_windows(text: str, budget: int = UNIT_BYTES) -> list[str]:
        if len(text.encode("utf-8")) <= budget:
            return [text]
        words = text.split()
        wins, cur, n = [], [], 0
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

    all_windows: list[str] = []
    counts: list[int] = []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(inputs)} paragraphs -> {len(all_windows)} windows", flush=True)

    prog = out_dir / "beam4_progress.jsonl"
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[gen] resuming with {len(saved)} saved windows", flush=True)

    missing = [i for i in range(len(all_windows)) if i not in saved]
    n_new = 0
    with torch.no_grad(), prog.open("a", encoding="utf-8") as prog_out:
        for bi in range(0, len(missing), 4):
            idxs = missing[bi : bi + 4]
            batch = [all_windows[i] for i in idxs]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1600
            ).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=3200, num_beams=4)
            batch_preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, pred in zip(idxs, batch_preds):
                prog_out.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
                saved[i] = pred
            n_new += len(idxs)
            if n_new % 80 == 0:
                prog_out.flush()
                checkpoints_volume.commit()
                print(f"[gen] {len(saved)}/{len(all_windows)} (committed)", flush=True)
    checkpoints_volume.commit()
    preds = [saved[i] for i in range(len(all_windows))]

    def project_haraqat(pred: str, text: str) -> str:
        pred_haraqat = [""]
        for ch in pred:
            if DIACRITICS_RE.match(ch):
                pred_haraqat[-1] += ch
            else:
                pred_haraqat.append("")
        pred_haraqat = pred_haraqat[1:]
        pred_letters = [c for c in pred if not DIACRITICS_RE.match(c)]
        text_letters = [c for c in text if not DIACRITICS_RE.match(c)]
        sm = SequenceMatcher(None, text_letters, pred_letters, autojunk=False)
        out = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    out.append(text_letters[i1 + k] + pred_haraqat[j1 + k])
            else:
                for k in range(i1, i2):
                    out.append(text_letters[k])
        return "".join(out)

    k = 0
    paragraphs = []
    for text, c in zip(inputs, counts):
        stitched = " ".join(preds[k : k + c])
        k += c
        paragraphs.append(project_haraqat(stitched, text))

    csv_path = Path("/tmp/sadeed_r6_beam4.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (out_dir / "sadeed_preds_beam4.csv").write_text(csv_path.read_text(), encoding="utf-8")
    checkpoints_volume.commit()

    import io
    import contextlib

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("\n===== r6 beam-4, windowed zero-skip =====", flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.report_errors_on_csv_file(
            str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
            gt_missing_diacritic_is_error=False)
    report = buf.getvalue()
    print(report, flush=True)
    (out_dir / "beam4_report.txt").write_text(report, encoding="utf-8")

    der_line = next((l for l in report.splitlines() if "DER" in l), "")
    result = {"greedy_der": GREEDY_DER, "beam": 4, "report_line": der_line.strip()}
    (out_dir / "beam4_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    done_marker.touch()
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(evaluate.remote(), indent=2))

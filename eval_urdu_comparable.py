"""Urdu comparable eval — settle urd-diac-1.0 vs d2 on ONE harness.

urd-diac-1.0 (urdu_diacrit/run-001, ByT5-small) shipped with 3.74 CER
on urdu-diacrit/test.jsonl; d2 (rababa_urdu_byt5/run-002-d2,
ByT5-base) measured 5.77 within its own lineage. Different lineages,
possibly different test files — this runs BOTH models greedy on the
SAME urdu-diacrit/test.jsonl so the manifest can name the real best.

Usage:
    modal run --detach eval_urdu_comparable.py
"""

from __future__ import annotations

import modal

urdu_volume = modal.Volume.from_name("urdu-diacrit-datasets", create_if_missing=False)
udiac_volume = modal.Volume.from_name("urdu-diacrit-checkpoints", create_if_missing=False)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "editdistance", "tqdm")
)

app = modal.App("rababa-urdu-comparable", image=image)

MODELS = {
    "urd_diacit_run001_shipped": "/volumes/udiac/urdu_diacrit/run-001/best",
    "d2_rababa_urdu_byt5": "/volumes/ckpts/rababa_urdu_byt5/run-002-d2/best",
}


@app.function(
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={
        "/datasets": urdu_volume,
        "/volumes/udiac": udiac_volume,
        "/volumes/ckpts": checkpoints_volume,
    },
)
def evaluate() -> dict:
    import json
    from pathlib import Path

    import editdistance
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    for v in (urdu_volume, udiac_volume, checkpoints_volume):
        v.reload()

    pairs = []
    with open("/datasets/urdu-diacrit/test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("src") and row.get("tgt"):
                    pairs.append((row["src"].strip(), row["tgt"].strip()))
    print(f"[data] {len(pairs)} test pairs", flush=True)

    results = {}
    for name, ckpt in MODELS.items():
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to("cuda")
        model.eval()
        preds: list[str] = []
        with torch.no_grad():
            for i in range(0, len(pairs), 64):
                chunk = pairs[i : i + 64]
                enc = tok([s for s, _ in chunk], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to("cuda")
                gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
                preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
        total_ed = total_len = exact = 0
        for (_, tgt), pred in zip(pairs, preds):
            total_ed += editdistance.eval(pred.strip(), tgt)
            total_len += len(tgt)
            exact += int(pred.strip() == tgt)
        res = {"cer": 100 * total_ed / max(1, total_len), "word_acc": 100 * exact / len(pairs)}
        results[name] = res
        print(f"[{name}] CER={res['cer']:.2f} word_acc={res['word_acc']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    return results


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(evaluate.remote(), indent=2))

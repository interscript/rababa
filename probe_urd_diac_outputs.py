"""Probe: what does the shipped urd-diac-1.0 checkpoint actually emit?

The comparable eval scored it CER 124.58 / word_acc 0.00 on
urdu-diacrit/test.jsonl under transformers 4.46.3 — print raw
(src, tgt, pred) triples to distinguish a broken artifact from a
harness/format mismatch.

Usage:
    modal run --detach probe_urd_diac_outputs.py
"""

from __future__ import annotations

import modal

urdu_volume = modal.Volume.from_name("urdu-diacrit-datasets", create_if_missing=False)
udiac_volume = modal.Volume.from_name("urdu-diacrit-checkpoints", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3")
)

app = modal.App("rababa-urd-probe", image=image)


@app.function(
    gpu="T4",
    timeout=20 * 60,
    volumes={"/datasets": urdu_volume, "/volumes/udiac": udiac_volume},
)
def probe() -> None:
    import json

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    urdu_volume.reload()
    udiac_volume.reload()

    pairs = []
    with open("/datasets/urdu-diacrit/test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("src") and row.get("tgt"):
                    pairs.append((row["src"].strip(), row["tgt"].strip()))
    sample = pairs[:5]

    ckpt = "/volumes/udiac/urdu_diacrit/run-001/best"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to("cuda")
    model.eval()
    enc = tok([s for s, _ in sample], return_tensors="pt", padding=True,
              truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
    preds = tok.batch_decode(gen, skip_special_tokens=True)
    for (src, tgt), pred in zip(sample, preds):
        print(f"SRC : {src[:80]}", flush=True)
        print(f"TGT : {tgt[:80]}", flush=True)
        print(f"PRED: {pred.strip()[:80]}", flush=True)
        print("---", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()

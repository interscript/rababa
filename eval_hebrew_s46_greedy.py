"""Greedy DER for s46 — the v1 runtime-path metric for heb-diac-1.1 metadata.

s46's verdict (16.43) is beam-4; the shipped runtime decodes greedy,
so the metadata needs the greedy number on the identical harness.
Resumable per example.

Usage:
    modal run --detach eval_hebrew_s46_greedy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

RUN = "rababa_hebrew/run-s46-phonikud-plus"
_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy>=1.26,<3", "tqdm>=4.66")
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-hebrew-s46-greedy", image=image)


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate() -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = str(Path("/checkpoints") / RUN / "run-002-gold-ft" / "best")
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to("cuda")
    tokenizer = ByT5Tokenizer.from_pretrained(ckpt)
    model.eval()

    test_path = Path(_find_nakdimon_root()) / "test.txt"
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        undiacritized = "".join(c for c in line if c not in _NIKUD_MARKS).strip()
        if 2 <= len(undiacritized) <= 512:
            examples.append((undiacritized, line))
    print(f"[eval] {len(examples)} examples", flush=True)

    prog = Path("/checkpoints") / RUN / "greedy_progress.jsonl"
    saved: dict[int, tuple[float, int]] = {}
    if prog.exists():
        for l in prog.read_text(encoding="utf-8").splitlines():
            if l.strip():
                row = json.loads(l)
                saved[row["i"]] = (row["der"], row["n"])
        print(f"[eval] resuming with {len(saved)} done", flush=True)

    missing = [i for i in range(len(examples)) if i not in saved]
    n_new = 0
    with torch.no_grad(), prog.open("a", encoding="utf-8") as out:
        for bi in range(0, len(missing), 16):
            idxs = missing[bi : bi + 16]
            batch = [examples[i][0] for i in idxs]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to("cuda")
            gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, pred in zip(idxs, preds):
                der, n = seq2seq_der(pred.strip(), examples[i][1])
                saved[i] = (der, n)
                out.write(json.dumps({"i": i, "der": der, "n": n}) + "\n")
            n_new += len(idxs)
            if n_new % 320 == 0:
                out.flush()
                checkpoints_volume.commit()
                total = sum(d * n for d, n in saved.values())
                cnt = sum(n for _, n in saved.values())
                print(f"[eval] {len(saved)}/{len(examples)} DER={total/max(1,cnt):.4f} (committed)", flush=True)
    checkpoints_volume.commit()

    total_wrong = sum(d * n for d, n in saved.values())
    total_positions = sum(n for _, n in saved.values())
    der = total_wrong / max(1, total_positions)
    print(f"=== s46 greedy DER: {der:.4f} ({len(saved)} examples) ===", flush=True)
    result = {"der_greedy": der, "n": len(saved), "checkpoint": ckpt,
              "der_beam4": 0.1643}
    (Path("/checkpoints") / RUN / "greedy_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(evaluate.remote(), indent=2))

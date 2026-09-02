"""Evaluate Hebrew v4 with beam=4, standard DER, 90-min timeout."""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "rababa"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.40,<5",
        "sentencepiece",
        "protobuf",
        "accelerate>=1.1.0",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name=f"{APP_NAME}-hebrew-v4-beam4", image=image)

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


@app.function(
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate_v4_beam4(checkpoint: str = "/checkpoints/rababa_hebrew_byt5_v4/run-001/best") -> dict:
    """Beam=4 eval with v2-compatible input (nikud stripped, teamim kept)."""
    import torch
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root

    checkpoints_volume.reload()
    datasets_volume.reload()

    device = torch.device("cuda")
    print(f"[v4-b4] loading {checkpoint}", flush=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
    tokenizer = ByT5Tokenizer.from_pretrained(checkpoint)
    model.eval()

    test_path = Path(_find_nakdimon_root()) / "test.txt"
    print(f"[v4-b4] test: {test_path}", flush=True)

    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        undiacritized = "".join(c for c in line if c not in _NIKUD_MARKS)
        undiacritized = undiacritized.strip()
        if len(undiacritized) >= 2 and len(undiacritized) <= 512:
            examples.append((undiacritized, line))

    print(f"[v4-b4] test examples: {len(examples)}", flush=True)

    total_wrong = 0
    total_positions = 0
    total_n = 0
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            src = [s for s, _ in batch]
            gold = [g for _, g in batch]
            enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for pred, g in zip(preds, gold):
                der, n = seq2seq_der(pred, g)
                total_wrong += int(der * n)
                total_positions += n
                total_n += 1

            if i % 320 == 0 and i > 0:
                der = total_wrong / max(1, total_positions)
                print(f"  [{i}/{len(examples)}] DER={der:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {"der": der, "n_examples": total_n, "checkpoint": checkpoint, "num_beams": 4}
    print(f"=== Hebrew v4 DER (beam=4): {der:.4f} ({total_n} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main(checkpoint: str = "/checkpoints/rababa_hebrew_byt5_v4/run-001/best"):
    result = evaluate_v4_beam4.remote(checkpoint=checkpoint)
    print(json.dumps(result, indent=2, ensure_ascii=False))

"""Evaluate Hebrew v3 with standard DER calculation, fast (num_beams=1)."""

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

app = modal.App(name=f"{APP_NAME}-hebrew-v3-fast-eval", image=image)


@app.function(
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate_v3_fast(checkpoint: str = "/checkpoints/rababa_hebrew_byt5_v3/run-001/best") -> dict:
    """Fast eval: num_beams=1, standard DER via rababa.seq2seq_der."""
    import torch
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root
    from torch.utils.data import Dataset, DataLoader
    from transformers import DataCollatorForSeq2Seq

    checkpoints_volume.reload()
    datasets_volume.reload()

    device = torch.device("cuda")
    print(f"[v3-fast] loading {checkpoint}", flush=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
    tokenizer = ByT5Tokenizer.from_pretrained(checkpoint)
    model.eval()

    # Load test set — use the MODERN Nakdimon test (matches v2 eval)
    nakdimon_root = Path(_find_nakdimon_root())
    test_path = nakdimon_root / "test.txt"
    print(f"[v3-fast] test: {test_path}", flush=True)

    # Parse test file: each line is a diacritized Hebrew sentence
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip nikud to get undiacritized input
        undiacritized = "".join(c for c in line if not ("֑" <= c <= "ׇ"))
        if undiacritized.strip():
            examples.append((undiacritized.strip(), line))

    print(f"[v3-fast] test examples: {len(examples)}", flush=True)

    total_wrong = 0
    total_positions = 0
    total_n = 0
    batch_size = 8  # small batch for long Biblical text

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            src = [s for s, _ in batch]
            gold = [g for _, g in batch]
            enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for pred, g in zip(preds, gold):
                der, n = seq2seq_der(pred, g)
                total_wrong += int(der * n)
                total_positions += n
                total_n += 1

            if i == 0:
                for j in range(min(2, len(batch))):
                    print(f"--- Example {i+j} ---", flush=True)
                    print(f"  in:   {src[j][:120]}", flush=True)
                    print(f"  pred: {preds[j][:120]}", flush=True)
                    print(f"  gold: {gold[j][:120]}", flush=True)

            if i % 320 == 0 and i > 0:
                der = total_wrong / max(1, total_positions)
                print(f"  [{i}/{len(examples)}] DER={der:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {
        "der": der,
        "n_examples": total_n,
        "checkpoint": checkpoint,
    }
    print(f"=== Hebrew v3 DER (standard): {der:.4f} ({total_n} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main(checkpoint: str = "/checkpoints/rababa_hebrew_byt5_v3/run-001/best"):
    result = evaluate_v3_fast.remote(checkpoint=checkpoint)
    print(json.dumps(result, indent=2, ensure_ascii=False))

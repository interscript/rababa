"""Expand Hebrew training data by distilling DictaBERT on Hebrew Wikipedia.

Key insight: DictaBERT is SOTA on modern Hebrew. Our test set is Rabbinic,
but by expanding training data with DictaBERT-labeled Wikipedia text, ByT5
can learn more Hebrew patterns. More data = better ByT5.

Process:
1. Load DictaBERT-menaked (transformers 4.38, confirmed working)
2. Run predict() on undiacritized Hebrew Wikipedia (hewiki/train.txt)
3. Save distilled predictions as new training data
4. Combine with existing gold corpus → expanded training set
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers==4.38.0",
        "huggingface_hub>=0.20,<0.25",
        "sentencepiece>=0.2",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "omegaconf>=2.3,<3",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .add_local_dir("configs", "/opt/rababa/configs", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name="rababa", image=image)


@app.function(
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def distill_hewiki() -> dict:
    """Run DictaBERT on Hebrew Wikipedia to expand training data."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from pathlib import Path as _P

    model_name = "dicta-il/dictabert-large-char-menaked"
    print(f"Loading {model_name}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # Load Hebrew Wikipedia
    hewiki_path = _P("/datasets/hewiki/train.txt")
    lines = []
    for line in hewiki_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if len(line) < 10 or len(line) > 200:
            continue
        lines.append(line)

    print(f"Hebrew Wikipedia lines: {len(lines)}", flush=True)

    # Subsample to 50K for reasonable distillation time
    max_lines = 50000
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        print(f"Subsampled to {len(lines)} lines", flush=True)

    # Distill
    out_dir = _P("/datasets/hebrew-dictabert-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.txt"

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, src in enumerate(lines):
            try:
                result = model.predict([src], tokenizer)
                pred = result[0] if result and result[0] else src
            except Exception:
                pred = src

            if pred.strip():
                f.write(pred.strip() + "\n")
                count += 1

            if i % 1000 == 0:
                print(f"[distill] {i}/{len(lines)}", flush=True)

    print(f"Distilled {count} lines → {out_path}", flush=True)
    datasets_volume.commit()

    return {"distilled_path": str(out_path), "count": count, "source": "hewiki"}


@app.local_entrypoint()
def main():
    result = distill_hewiki.remote()
    print(json.dumps(result, indent=2, default=str))

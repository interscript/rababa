"""Distill DictaBERT predictions into training data for ByT5.

Strategy:
1. Load DictaBERT-menaked (transformers 4.38, confirmed working)
2. Run predict() on undiacritized Hebrew training data
3. Save distilled predictions alongside gold labels
4. Train ByT5 on gold + distilled data

This transfers DictaBERT's Hebrew knowledge to ByT5 without needing
to modify DictaBERT's custom training code.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

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
def distill_dictabert() -> dict:
    """Run DictaBERT on training data, save distilled predictions."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from rababa.datasets import _find_nakdimon_root
    from rababa.evaluate import _NIQQUD_MARKS
    from pathlib import Path as _P

    model_name = "dicta-il/dictabert-large-char-menaked"
    print(f"Loading {model_name} with transformers 4.38...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # Load training data
    data_root = _P(_find_nakdimon_root())
    train_path = data_root / "train.txt"

    def _strip_diacritics(text):
        return "".join(c for c in text if c not in _NIQQUD_MARKS)

    # Read undiacritized training lines
    lines = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
        diacritized = line.strip()
        if not diacritized:
            continue
        undiacritized = _strip_diacritics(diacritized)
        if len(undiacritized) < 2 or len(undiacritized) > 200:
            continue
        lines.append((undiacritized, diacritized))

    print(f"Training lines: {len(lines)}", flush=True)

    # Run DictaBERT on each line
    distilled_dir = _P("/datasets/hebrew-distilled-v2")
    distilled_dir.mkdir(parents=True, exist_ok=True)
    distilled_train = distilled_dir / "train.txt"

    count = 0
    with distilled_train.open("w", encoding="utf-8") as f:
        for i, (src, gold) in enumerate(lines):
            try:
                result = model.predict([src], tokenizer)
                pred = result[0] if result and result[0] else src
            except Exception:
                pred = src
            f.write(pred + "\n")
            count += 1

            if i % 1000 == 0:
                print(f"[distill] {i}/{len(lines)}", flush=True)

    print(f"Distilled {count} lines → {distilled_train}", flush=True)

    # Also distill val and test
    for split in ("val", "test"):
        split_path = data_root / f"{split}.txt"
        if not split_path.is_file():
            continue
        out_path = distilled_dir / f"{split}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for line in split_path.read_text(encoding="utf-8").splitlines():
                diacritized = line.strip()
                if not diacritized:
                    continue
                undiacritized = _strip_diacritics(diacritized)
                if len(undiacritized) < 2 or len(undiacritized) > 200:
                    f.write(diacritized + "\n")
                    continue
                try:
                    result = model.predict([undiacritized], tokenizer)
                    pred = result[0] if result and result[0] else undiacritized
                except Exception:
                    pred = undiacritized
                f.write(pred + "\n")
        print(f"Distilled {split} → {out_path}", flush=True)

    datasets_volume.commit()
    return {"distilled_train": str(distilled_train), "count": count}


@app.local_entrypoint()
def main():
    result = distill_dictabert.remote()
    print(json.dumps(result, indent=2, default=str))

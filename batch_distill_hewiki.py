"""Batch DictaBERT distillation on Hebrew Wikipedia.

DictaBERT's predict() takes a list — try batching 32 texts per call
to speed up from ~7s/example to ~0.2s/example (32x speedup).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

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
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name="rababa", image=image)


@app.function(
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def batch_distill_hewiki() -> dict:
    """Batch distill Hebrew Wikipedia with DictaBERT."""
    import time
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

    max_lines = 50000
    lines = lines[:max_lines]
    print(f"Processing {len(lines)} Hebrew Wikipedia lines", flush=True)

    # Test batch prediction speed
    print("Testing batch speed...", flush=True)
    test_batch = lines[:32]
    t0 = time.time()
    result = model.predict(test_batch, tokenizer)
    batch_time = time.time() - t0
    print(f"Batch of 32: {batch_time:.1f}s ({batch_time/32:.2f}s/example)", flush=True)
    print(f"Sample output: {result[0][:60] if result else 'EMPTY'}", flush=True)

    # If batch works, process all lines in batches of 32
    out_dir = _P("/datasets/hebrew-dictabert-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.txt"

    batch_size = 32
    count = 0
    total_batches = (len(lines) + batch_size - 1) // batch_size
    t_start = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            try:
                predictions = model.predict(batch, tokenizer)
            except Exception as e:
                print(f"Batch {i//batch_size} error: {e}", flush=True)
                predictions = batch  # fallback to input

            for pred in predictions:
                if pred and pred.strip():
                    f.write(pred.strip() + "\n")
                    count += 1

            batch_num = i // batch_size + 1
            if batch_num % 50 == 0:
                elapsed = time.time() - t_start
                rate = count / max(1, elapsed)
                eta = (len(lines) - count) / max(1, rate)
                print(f"  [{batch_num}/{total_batches}] {count} lines, "
                      f"{rate:.1f}/s, ETA {eta/60:.0f}min", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone! {count} lines in {elapsed/60:.1f}min "
          f"({count/max(1,elapsed):.1f} lines/sec)", flush=True)

    datasets_volume.commit()
    return {
        "distilled_path": str(out_path),
        "count": count,
        "elapsed_min": elapsed / 60,
        "rate_per_sec": count / max(1, elapsed),
    }


@app.local_entrypoint()
def main():
    result = batch_distill_hewiki.remote()
    print(json.dumps(result, indent=2, default=str))

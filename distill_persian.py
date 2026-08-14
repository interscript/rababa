"""Cross-lingual distillation: use Arabic model to label Persian/Urdu text.

Since Persian/Urdu share the Arabic script and harakat system, our Arabic
model (1.30% DER) can produce reasonable diacritization on Persian/Urdu text.
This creates training data where none existed before.

Pipeline:
1. Download Ganjoor Persian poetry (undiacritized)
2. Run Arabic rababa model to generate diacritized predictions
3. Save as Persian training corpus
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
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .add_local_dir("configs", "/opt/rababa/configs", copy=True)
    .add_local_dir("test-datasets", "/opt/rababa/test-datasets", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name="rababa", image=image)


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def distill_persian() -> dict:
    """Download Persian text + use Arabic model to diacritize it."""
    import torch
    import subprocess
    from pathlib import Path as _P
    from rababa.models.base import build_model
    from rababa.config import load_task_config, to_dict
    from rababa.encoder import ArabicEncoder

    # 1. Download Ganjoor Persian poetry
    print("=== Downloading Ganjoor Persian poetry ===", flush=True)
    ganjoor_dir = _P("/tmp/persian-poetry")
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/aghasemi/ChronologicalPersianPoetryDataset.git",
         str(ganjoor_dir)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        # Try alternative source
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Mohampouraz/Persian-poetry.git",
             str(ganjoor_dir)],
            capture_output=True, text=True, timeout=120
        )

    if result.returncode != 0:
        return {"error": f"Failed to clone: {result.stderr[:200]}"}

    # Collect Persian text lines
    persian_lines = []
    for txt_file in ganjoor_dir.rglob("*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if len(line) < 10 or len(line) > 200:
                    continue
                persian_lines.append(line)
        except Exception:
            continue

    print(f"Collected {len(persian_lines)} Persian lines", flush=True)

    if len(persian_lines) > 100000:
        persian_lines = persian_lines[:100000]
        print(f"Subsampled to {len(persian_lines)} lines", flush=True)

    # 2. Load Arabic model
    print("\n=== Loading Arabic model ===", flush=True)
    ckpt_path = "/checkpoints/rababa_arabic_v2/run-001/best.pt"
    if not _P(ckpt_path).exists():
        # Try original rababa_arabic checkpoint
        ckpt_path = "/checkpoints/rababa_arabic/run-001/best.pt"

    cfg = load_task_config("rababa_arabic")
    cfg_dict = to_dict(cfg)
    device = torch.device("cuda")

    model = build_model(cfg_dict).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Arabic model loaded from {ckpt_path}", flush=True)

    # 3. Diacritize Persian text using Arabic model
    print("\n=== Diacritizing Persian text with Arabic model ===", flush=True)
    encoder = ArabicEncoder(cleaner="arabic")

    haraqat_map = {0: ""}
    from rababa.constants import HARAQAT_LIST
    for i, h in enumerate(HARAQAT_LIST):
        haraqat_map[i] = h

    out_dir = _P("/datasets/persian-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.txt"

    count = 0
    batch_size = 64
    max_len = 200

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(persian_lines), batch_size):
            batch_lines = persian_lines[i:i+batch_size]
            batch_src = []
            for line in batch_lines:
                src_ids = encoder.encode(line)[:max_len]
                batch_src.append(src_ids)

            # Pad
            T = max(len(s) for s in batch_src)
            B = len(batch_src)
            src_tensor = torch.zeros(B, T, dtype=torch.long, device=device)
            lengths = torch.zeros(B, dtype=torch.long, device=device)
            for j, s in enumerate(batch_src):
                src_tensor[j, :len(s)] = torch.tensor(s)
                lengths[j] = len(s)

            with torch.no_grad():
                outputs = model.forward_heads(src_tensor, lengths)
                logits = outputs[0]  # haraqat head

            preds = logits.argmax(dim=-1)

            for j in range(B):
                src_ids = batch_src[j]
                pred_ids = preds[j][:len(src_ids)].tolist()
                # Reconstruct diacritized text
                diacritized = []
                for char_idx, (char_id, pred_id) in enumerate(zip(src_ids, pred_ids)):
                    char = encoder.id_to_char.get(char_id, "")
                    haraka = haraqat_map.get(pred_id, "")
                    diacritized.append(char + haraka)
                result_text = "".join(diacritized)
                f.write(result_text + "\n")
                count += 1

            if i % 5000 == 0 and i > 0:
                print(f"  [{i}/{len(persian_lines)}] distilled {count} lines", flush=True)

    print(f"\nDistilled {count} Persian lines → {out_path}", flush=True)

    # Split into train/val/test
    import random
    random.seed(42)
    with out_path.open("r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]
    random.shuffle(all_lines)

    n_test = max(500, len(all_lines) // 20)
    n_val = max(500, len(all_lines) // 20)
    for split, data in [
        ("test", all_lines[:n_test]),
        ("val", all_lines[n_test:n_test+n_val]),
        ("train", all_lines[n_test+n_val:]),
    ]:
        (out_dir / f"{split}.txt").write_text("\n".join(data) + "\n", encoding="utf-8")
        print(f"  {split}: {len(data)} lines", flush=True)

    datasets_volume.commit()

    return {
        "total_lines": count,
        "train": len(all_lines) - n_test - n_val,
        "val": n_val,
        "test": n_test,
        "output_dir": str(out_dir),
    }


@app.local_entrypoint()
def main():
    result = distill_persian.remote()
    print(json.dumps(result, indent=2, default=str))

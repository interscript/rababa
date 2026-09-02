"""Cross-lingual distillation for Urdu: use Arabic model to label Urdu text.

Same approach as Persian: our Arabic model (1.30% DER) diacritizes Urdu
text from Wikipedia/news, creating training data where none existed.
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
        "torch>=2.4,<3", "numpy>=1.26,<3", "omegaconf>=2.3,<3",
        "tqdm>=4.66", "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .add_local_dir("configs", "/opt/rababa/configs", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name="rababa", image=image)


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def distill_urdu() -> dict:
    """Download Urdu text + use Arabic model to diacritize it."""
    import torch
    import subprocess
    from pathlib import Path as _P
    from rababa.models.base import build_model
    from rababa.config import load_task_config, to_dict
    from rababa.encoder import ArabicEncoder

    # 1. Download Urdu text (multiple sources)
    print("=== Downloading Urdu text ===", flush=True)

    urdu_lines = []

    # Source 1: Urdu poetry from GitHub
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/harismuneer/Urdu-Text-Data-Set.git",
         "/tmp/urdu-text"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        urdu_dir = _P("/tmp/urdu-text")
        for txt_file in urdu_dir.rglob("*.txt"):
            try:
                for line in txt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if 10 <= len(line) <= 200:
                        urdu_lines.append(line)
            except Exception:
                continue
        print(f"  Urdu poetry: {len(urdu_lines)} lines", flush=True)

    # Source 2: Quran Urdu translation (fully diacritized)
    result2 = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/Jalees-Project/quran-urdu.git",
         "/tmp/quran-urdu"],
        capture_output=True, text=True, timeout=60
    )
    if result2.returncode == 0:
        quran_dir = _P("/tmp/quran-urdi")
        for txt_file in quran_dir.rglob("*.txt"):
            try:
                for line in txt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if 10 <= len(line) <= 200:
                        urdu_lines.append(line)
            except Exception:
                continue

    # Source 3: Generate Urdu text from available corpora on HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("wikipedia", "20220301.ur", split="train", streaming=True)
        for i, ex in enumerate(ds):
            text = ex.get("text", "")
            for line in text.splitlines():
                line = line.strip()
                if 10 <= len(line) <= 200:
                    urdu_lines.append(line)
            if len(urdu_lines) >= 50000:
                break
    except Exception as e:
        print(f"  HuggingFace Wikipedia: {e}", flush=True)

    # Deduplicate
    urdu_lines = list(set(urdu_lines))
    print(f"Total unique Urdu lines: {len(urdu_lines)}", flush=True)

    if len(urdu_lines) < 100:
        return {"error": f"Only {len(urdu_lines)} Urdu lines collected. Need more sources."}

    # Subsample
    max_lines = 100000
    if len(urdu_lines) > max_lines:
        urdu_lines = urdu_lines[:max_lines]

    # 2. Load Arabic model
    print("\n=== Loading Arabic model ===", flush=True)
    ckpt_path = "/checkpoints/rababa_arabic_v2/run-001/best.pt"
    if not _P(ckpt_path).exists():
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

    # 3. Diacritize
    print("\n=== Diacritizing Urdu text ===", flush=True)
    encoder = ArabicEncoder(cleaner="arabic")
    from rababa.constants import HARAQAT_LIST
    haraqat_map = {i: h for i, h in enumerate(HARAQAT_LIST)}

    out_dir = _P("/datasets/urdu-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    batch_size = 64
    max_len = 200
    all_diacritized = []

    for i in range(0, len(urdu_lines), batch_size):
        batch_lines = urdu_lines[i:i+batch_size]
        batch_src = [encoder.encode(l)[:max_len] for l in batch_lines]

        T = max(len(s) for s in batch_src)
        B = len(batch_src)
        src_tensor = torch.zeros(B, T, dtype=torch.long, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        for j, s in enumerate(batch_src):
            src_tensor[j, :len(s)] = torch.tensor(s)
            lengths[j] = len(s)

        with torch.no_grad():
            outputs = model.forward_heads(src_tensor, lengths)
            preds = outputs[0].argmax(dim=-1)

        for j in range(B):
            src_ids = batch_src[j]
            pred_ids = preds[j][:len(src_ids)].tolist()
            diacritized = []
            for char_id, pred_id in zip(src_ids, pred_ids):
                char = encoder.id_to_char.get(char_id, "")
                haraka = haraqat_map.get(pred_id, "")
                diacritized.append(char + haraka)
            all_diacritized.append("".join(diacritized))
            count += 1

        if i % 5000 == 0 and i > 0:
            print(f"  [{i}/{len(urdu_lines)}] distilled {count}", flush=True)

    # Split
    import random
    random.seed(42)
    random.shuffle(all_diacritized)
    n_test = max(500, len(all_diacritized) // 20)
    n_val = max(500, len(all_diacritized) // 20)
    for split, data in [
        ("test", all_diacritized[:n_test]),
        ("val", all_diacritized[n_test:n_test+n_val]),
        ("train", all_diacritized[n_test+n_val:]),
    ]:
        (out_dir / f"{split}.txt").write_text("\n".join(data) + "\n", encoding="utf-8")
        print(f"  {split}: {len(data)} lines", flush=True)

    datasets_volume.commit()
    return {"total": count, "output_dir": str(out_dir)}


@app.local_entrypoint()
def main():
    result = distill_urdu.remote()
    print(json.dumps(result, indent=2, default=str))

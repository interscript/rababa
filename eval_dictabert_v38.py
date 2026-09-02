"""Evaluate DictaBERT-menaked with older transformers version.

The custom code for dictabert-large-char-menaked produces garbled output
with transformers >= 4.46. Trying transformers==4.38.0 which was current
when the model was released.
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
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_dictabert_v38() -> dict:
    """Load dictabert-large-char-menaked with transformers 4.38 and test."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    import transformers

    print(f"transformers version: {transformers.__version__}", flush=True)

    model_name = "dicta-il/dictabert-large-char-menaked"
    print(f"Loading {model_name}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print(f"Model type: {type(model).__name__}", flush=True)

    # Smoke test with model card example
    sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
    print(f"\n=== Smoke test ===", flush=True)
    print(f"Input:  {sentence}", flush=True)

    result = model.predict([sentence], tokenizer)
    pred = result[0] if result else ""
    print(f"Output: {pred}", flush=True)

    # Also try with mark_matres_lectionis
    result2 = model.predict([sentence], tokenizer, mark_matres_lectionis='*')
    pred2 = result2[0] if result2 else ""
    print(f"Output (mrl=*): {pred2}", flush=True)

    # Load test data
    from rababa.datasets import _find_nakdimon_root
    from rababa.evaluate import seq2seq_der, _NIQQUD_MARKS
    from pathlib import Path as _P

    data_root = _P(_find_nakdimon_root())
    test_path = data_root / "test.txt"

    def _strip_diacritics(text):
        return "".join(c for c in text if c not in _NIQQUD_MARKS)

    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        diacritized = line.strip()
        if not diacritized:
            continue
        undiacritized = _strip_diacritics(diacritized)
        if len(undiacritized) < 2 or len(undiacritized) > 200:
            continue
        examples.append((undiacritized, diacritized))

    # Subsample to 1000 for quick evaluation
    examples = examples[:1000]
    print(f"\nTest examples (subsampled): {len(examples)}", flush=True)

    total_wrong = 0
    total_positions = 0
    total_n = 0

    for i, (src, gold) in enumerate(examples):
        try:
            result = model.predict([src], tokenizer)
            pred = result[0] if result else src
        except Exception as e:
            pred = src

        der, n = seq2seq_der(pred, gold)
        total_wrong += int(der * n)
        total_positions += n
        total_n += 1

        if i < 3:
            print(f"\n--- Example {i} ---", flush=True)
            print(f"  input: {src[:60]}", flush=True)
            print(f"  pred:  {pred[:60]}", flush=True)
            print(f"  gold:  {gold[:60]}", flush=True)
            print(f"  DER:   {der:.4f}", flush=True)

        if i % 200 == 0 and i > 0:
            agg = total_wrong / max(1, total_positions)
            print(f"  [{i}/{len(examples)}] DER={agg:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {
        "model": model_name,
        "transformers_version": transformers.__version__,
        "der": der,
        "n_examples": total_n,
    }
    print(f"\n=== DictaBERT DER (transformers {transformers.__version__}): {der:.4f} ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_dictabert_v38.remote()
    print(json.dumps(result, indent=2, default=str))

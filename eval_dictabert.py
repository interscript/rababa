"""Evaluate DictaBERT (dicta-il/dictabert-large-char-menaked) on Hebrew test set.

This is a ready-made Hebrew diacritization model from the Dicta team —
the same team behind the SOTA Dicta Nakdan commercial system.

The model uses character-level BERT with custom code for nikud prediction.
We evaluate it on our Nakdimon test split to see if it beats our ByT5-small
(16.97% DER).
"""

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
        "transformers>=4.46",
        "huggingface_hub>=0.26",
        "sentencepiece>=0.2",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .add_local_dir("configs", "/opt/rababa/configs", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_dictabert() -> dict:
    """Load dictabert-large-char-menaked and evaluate DER on Hebrew test set."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "dicta-il/dictabert-large-char-menaked"
    print(f"Loading {model_name}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print(f"Model loaded. Type: {type(model).__name__}", flush=True)

    # Test with the model card's example sentence first
    test_sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
    print(f"\n=== DictaBERT smoke test ===", flush=True)
    print(f"Input:  {test_sentence}", flush=True)
    result = model.predict([test_sentence], tokenizer)
    print(f"Output: {result[0] if result else 'EMPTY'}", flush=True)

    # Load test data from the combined Hebrew corpus
    from rababa.datasets import _find_nakdimon_root
    from rababa.evaluate import seq2seq_der, _NIQQUD_MARKS
    from pathlib import Path as _P

    data_root = _P(_find_nakdimon_root())
    test_path = data_root / "test.txt"

    def _strip_diacritics(text):
        return "".join(c for c in text if c not in _NIQQUD_MARKS)

    # Load test examples
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        diacritized = line.strip()
        if not diacritized:
            continue
        undiacritized = _strip_diacritics(diacritized)
        if len(undiacritized) < 2:
            continue
        examples.append((undiacritized, diacritized))

    print(f"Test examples: {len(examples)}", flush=True)

    # Check if model has a predict method (custom code)
    has_predict = hasattr(model, "predict") or hasattr(model, "nikud")
    print(f"Has predict method: {has_predict}", flush=True)
    print(f"Model methods: {[m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))][:10]}", flush=True)

    total_wrong = 0
    total_positions = 0
    total_n = 0

    with torch.no_grad():
        for i in range(0, len(examples), 1):
            src, gold = examples[i]
            try:
                predictions = model.predict([src], tokenizer, mark_matres_lectionis='*')
                pred = predictions[0] if predictions else src
            except Exception as e:
                if i == 0:
                    print(f"predict error: {e}", flush=True)
                pred = src

            der, n = seq2seq_der(pred, gold)
            total_wrong += int(der * n)
            total_positions += n
            total_n += 1

            if i < 5:
                print(f"\n--- Example {i} ---", flush=True)
                print(f"  input: {src[:60]}", flush=True)
                print(f"  pred:  {pred[:60]}", flush=True)
                print(f"  gold:  {gold[:60]}", flush=True)
                print(f"  DER:   {der:.4f}", flush=True)

            if i % 500 == 0 and i > 0:
                agg_der = total_wrong / max(1, total_positions)
                print(f"\n  [{i}/{len(examples)}] running DER={agg_der:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {
        "model": model_name,
        "der": der,
        "n_examples": total_n,
    }
    print(f"=== DictaBERT DER: {der:.4f} ({total_n} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_dictabert.remote()
    print(json.dumps(result, indent=2))

"""Evaluate NadavShaked/D_Nikud — the SOTA Hebrew diacritization model.

D-Nikud achieves 98.26% DEC (Decision Accuracy) using TavBERT + BiLSTM.
Available on HuggingFace as a RoBERTa model with NO custom code.
If this works, Hebrew DER drops from 17% to ~2% instantly.
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
        "transformers>=4.40",
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

app = modal.App(name="rababa", image=image)


@app.function(
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_dnikud() -> dict:
    """Load D-Nikud model and evaluate on our Hebrew test set."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from rababa.datasets import _find_nakdimon_root
    from rababa.evaluate import seq2seq_der, _NIQQUD_MARKS
    from pathlib import Path as _P

    model_name = "NadavShaked/D_Nikud"
    print(f"Loading {model_name}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to("cuda")
    model.eval()

    print(f"Model type: {type(model).__name__}", flush=True)
    print(f"Num labels: {model.config.num_labels}", flush=True)
    print(f"Labels: {model.config.id2label if hasattr(model.config, 'id2label') else 'N/A'}", flush=True)

    # Smoke test
    test = "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת"
    print(f"\n=== Smoke test ===", flush=True)
    print(f"Input: {test}", flush=True)

    inputs = tokenizer(test, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_ids = logits.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [model.config.id2label.get(i.item(), '?') for i in pred_ids[0]]

    print(f"Tokens: {tokens[:20]}", flush=True)
    print(f"Labels: {labels[:20]}", flush=True)

    # Load test data
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

    examples = examples[:1000]
    print(f"\nTest examples (subsampled): {len(examples)}", flush=True)

    total_wrong = 0
    total_positions = 0
    total_n = 0

    for i, (src, gold) in enumerate(examples):
        try:
            inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            pred_ids = outputs.logits.argmax(dim=-1)

            # Reconstruct diacritized text from predictions
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            labels = [model.config.id2label.get(idx.item(), '') for idx in pred_ids[0]]

            # Merge tokens with labels to reconstruct text
            pred_parts = []
            for tok, label in zip(tokens, labels):
                if tok in ['<s>', '</s>', '<pad>', '<mask>']:
                    continue
                # Clean token (remove RoBERTa prefix)
                clean = tok.lstrip('Ġ').lstrip('ּ')
                if clean:
                    # The label might be the diacritized form
                    if label and label != '0' and label != 'O':
                        pred_parts.append(label)
                    else:
                        pred_parts.append(clean)
            pred = ' '.join(pred_parts)
        except Exception as e:
            if i == 0:
                print(f"Prediction error: {e}", flush=True)
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

        if i % 200 == 0 and i > 0:
            agg = total_wrong / max(1, total_positions)
            print(f"  [{i}/{len(examples)}] DER={agg:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {
        "model": model_name,
        "der": der,
        "n_examples": total_n,
    }
    print(f"\n=== D-Nikud DER: {der:.4f} ({total_n} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_dnikud.remote()
    print(json.dumps(result, indent=2, default=str))

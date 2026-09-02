"""Evaluate DictaBERT-large-char-menaked directly on our Hebrew test set.

DictaBERT is the SOTA Hebrew diacritization model. It uses a custom
BertForDiacritization head. Requires transformers==4.38.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "rababa"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.2,<2.4",
        "transformers==4.38.0",
        "sentencepiece",
        "protobuf",
        "numpy>=1.26,<2",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name=f"{APP_NAME}-dictabert-eval", image=image)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_dictabert() -> dict:
    """Load DictaBERT and evaluate on our Hebrew test set."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("[dictabert] loading model...", flush=True)
    model_name = "dicta-il/dictabert-large-char-menaked"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    model.eval()
    print("[dictabert] model loaded", flush=True)

    # Smoke test
    test = "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת"
    pred = model.predict([test], tokenizer)
    print(f"[dictabert] smoke test:", flush=True)
    print(f"  in:  {test}", flush=True)
    print(f"  out: {pred[0]}", flush=True)

    # Load test set
    from rababa.datasets import _find_nakdimon_root
    test_path = Path(_find_nakdimon_root()) / "test.txt"
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # undiacritize: strip haraqat
        undiacritized = _strip_haraqat(line)
        if undiacritized and line:
            examples.append((undiacritized, line))
    print(f"[dictabert] test examples: {len(examples)}", flush=True)

    # Process in batches (batch_size=8 to avoid OOM on long Biblical verses)
    batch_size = 8
    total_wrong = 0
    total_positions = 0
    total_n = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        undiacritized = [s for s, _ in batch]
        gold = [g for _, g in batch]
        try:
            preds = model.predict(undiacritized, tokenizer)
        except Exception as e:
            print(f"  batch {i} error: {e}", flush=True)
            continue

        for pred, g in zip(preds, gold):
            wrong, total = _compare_diacritized(pred, g)
            total_wrong += wrong
            total_positions += total
            total_n += 1

        if i < 3 * batch_size and i == 0:
            for j in range(min(3, len(batch))):
                print(f"--- Example {i+j} ---", flush=True)
                print(f"  in:   {undiacritized[j]}", flush=True)
                print(f"  pred: {preds[j]}", flush=True)
                print(f"  gold: {gold[j]}", flush=True)

        if i % 320 == 0 and i > 0:
            der = total_wrong / max(1, total_positions)
            print(f"  [{i}/{len(examples)}] DER={der:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {
        "model": model_name,
        "der": der,
        "n_examples": total_n,
    }
    print(f"=== DictaBERT DER: {der:.4f} ({total_n} examples) ===", flush=True)
    return result


def _strip_haraqat(s: str) -> str:
    """Strip Hebrew diacritics (nikud) from a string."""
    # Hebrew nikud Unicode block: U+0591 to U+05C7
    out = []
    for c in s:
        if "֑" <= c <= "ׇ":
            continue
        out.append(c)
    return "".join(out)


def _compare_diacritized(pred: str, gold: str) -> tuple[int, int]:
    """Count wrong chars and total chars (only on consonant positions)."""
    # Walk consonant-by-consonant, compare the diacritics that follow.
    # Strip to consonants + their following diacritics.
    def _split(s):
        result = []
        cur_c = None
        cur_diacritics = []
        for c in s:
            if "֑" <= c <= "ׇ":
                cur_diacritics.append(c)
            else:
                if cur_c is not None:
                    result.append((cur_c, "".join(cur_diacritics)))
                cur_c = c
                cur_diacritics = []
        if cur_c is not None:
            result.append((cur_c, "".join(cur_diacritics)))
        return result

    p = _split(pred)
    g = _split(gold)
    if len(p) != len(g):
        return max(len(p), len(g)), max(len(p), len(g))
    wrong = sum(1 for a, b in zip(p, g) if a != b)
    return wrong, len(g)


@app.local_entrypoint()
def main():
    result = evaluate_dictabert.remote()
    print(json.dumps(result, indent=2, ensure_ascii=False))

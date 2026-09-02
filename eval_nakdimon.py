"""Evaluate Nakdimon's bundled ONNX model on our Hebrew test set.

Nakdimon ships a pre-trained ONNX model in the wheel. No training needed —
just install, predict, and compute DER.
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
        "transformers>=4.30",
        "sentencepiece>=0.2",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "nakdimon",
        "onnxruntime>=1.20",
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
)
def evaluate_nakdimon() -> dict:
    """Use Nakdimon's bundled model to predict diacritics on our test set."""
    import nakdimon
    from rababa.datasets import _find_nakdimon_root
    from rababa.evaluate import seq2seq_der, _NIQQUD_MARKS
    from pathlib import Path as _P

    # Explore Nakdimon API
    attrs = [a for a in dir(nakdimon) if not a.startswith('_')]
    print(f"Nakdimon attributes: {attrs}", flush=True)

    # The main function is likely 'diacritize'
    diacritize_fn = getattr(nakdimon, 'diacritize', None) or getattr(nakdimon, 'do_predict', None)
    if diacritize_fn is None:
        return {"error": f"No diacritize function found. Attrs: {attrs}"}
    print(f"Using: {diacritize_fn.__name__}", flush=True)

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
        if len(undiacritized) < 2:
            continue
        examples.append((undiacritized, diacritized))

    # Subsample for speed — first 1000 for quick estimate
    examples = examples[:1000]
    print(f"Test examples (subsampled): {len(examples)}", flush=True)

    # Smoke test
    test_input = "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת"
    print(f"\n=== Smoke test ===", flush=True)
    print(f"Input:  {test_input}", flush=True)
    try:
        result = diacritize_fn(test_input)
        print(f"Output: {result}", flush=True)
    except Exception as e:
        print(f"diacritize error: {e}", flush=True)
        import inspect
        print(f"Signature: {inspect.signature(diacritize_fn)}", flush=True)
        return {"error": str(e)}

    # Full evaluation
    total_wrong = 0
    total_positions = 0
    total_n = 0

    for i, (src, gold) in enumerate(examples):
        try:
            pred = diacritize_fn(src)
        except Exception:
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
            agg = total_wrong / max(1, total_positions)
            print(f"  [{i}/{len(examples)}] DER={agg:.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    result = {
        "model": "Nakdimon (bundled ONNX)",
        "der": der,
        "n_examples": total_n,
    }
    print(f"\n=== Nakdimon DER: {der:.4f} ({total_n} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_nakdimon.remote()
    print(json.dumps(result, indent=2, default=str))

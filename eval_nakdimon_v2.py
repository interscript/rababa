"""Evaluate Nakdimon using its diacritize function.

Nakdimon ships with a bundled ONNX model. The `diacritize` function
is the main entry point for diacritizing text.
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
    cpu=4,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def evaluate_nakdimon_v2() -> dict:
    """Use Nakdimon's diacritize function to predict on test set."""
    import nakdimon
    import inspect

    # Explore the diacritize function
    print(f"Nakdimon attributes: {[a for a in dir(nakdimon) if not a.startswith('_')]}", flush=True)

    diac_fn = getattr(nakdimon, 'diacritize', None)
    if diac_fn is None:
        # Try do_predict
        diac_fn = getattr(nakdimon, 'do_predict', None)

    if diac_fn is None:
        return {"error": "No diacritize or do_predict function found"}

    print(f"Using: {diac_fn.__name__}", flush=True)
    try:
        sig = inspect.signature(diac_fn)
        print(f"Signature: {sig}", flush=True)
    except Exception:
        pass

    # Try calling with a test sentence
    test = "בשנת 1948 השלים אפרים קישון את לימודיו"
    print(f"\n=== Smoke test ===", flush=True)
    print(f"Input: {test}", flush=True)

    # Try different calling conventions
    pred = None
    for attempt in [
        lambda: diac_fn(test),
        lambda: diac_fn(text=test),
        lambda: diac_fn([test]),
        lambda: diac_fn(model=nakdimon.MAIN_MODEL, text=test),
        lambda: diac_fn(nakdimon.MAIN_MODEL, test),
    ]:
        try:
            result = attempt()
            if isinstance(result, str):
                pred = result
            elif isinstance(result, list) and result:
                pred = result[0] if isinstance(result[0], str) else str(result[0])
            else:
                pred = str(result)
            print(f"Success with attempt! Output: {pred}", flush=True)
            break
        except Exception as e:
            print(f"  attempt failed: {e}", flush=True)

    if pred is None:
        # Check config for model loading hints
        cfg = getattr(nakdimon, 'config', None)
        if cfg:
            print(f"Config: {dir(cfg)}", flush=True)
        return {"error": "All calling conventions failed"}

    # Load test data
    from rababa.datasets import _find_nakdimon_root
    from rabba.evaluate import seq2seq_der, _NIQQUD_MARKS
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

    examples = examples[:500]
    print(f"\nTest examples (subsampled): {len(examples)}", flush=True)

    total_wrong = 0
    total_positions = 0
    total_n = 0

    for i, (src, gold) in enumerate(examples):
        try:
            result = diac_fn(src)
            pred = result if isinstance(result, str) else str(result)
        except Exception:
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

    der = total_wrong / max(1, total_positions)
    print(f"\n=== Nakdimon DER: {der:.4f} ({total_n} examples) ===", flush=True)
    return {"model": "Nakdimon", "der": der, "n_examples": total_n}


@app.local_entrypoint()
def main():
    result = evaluate_nakdimon_v2.remote()
    print(json.dumps(result, indent=2, default=str))

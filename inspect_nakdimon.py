"""Inspect and use Nakdimon's diacritize function correctly.

Previous attempts failed because we didn't understand the API.
This script first inspects the function source, then uses it correctly.
"""

from __future__ import annotations

import json
import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.30",
        "numpy>=1.26,<3",
        "nakdimon",
        "onnxruntime>=1.20",
        "omegaconf>=2.3,<3",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name="rababa", image=image)


@app.function(
    cpu=4,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def inspect_nakdimon() -> dict:
    """Inspect Nakdimon API, then evaluate on Hebrew test set."""
    import nakdimon
    import inspect

    # 1. Inspect the diacritize function
    diac_fn = nakdimon.diacritize
    print("=== diacritize source ===", flush=True)
    try:
        src = inspect.getsource(diac_fn)
        print(src[:3000], flush=True)
    except Exception as e:
        print(f"Can't get source: {e}", flush=True)

    print("\n=== Signature ===", flush=True)
    try:
        sig = inspect.signature(diac_fn)
        print(f"diacritize{sig}", flush=True)
    except Exception:
        pass

    # 2. Check MAIN_MODEL and config
    print(f"\nMAIN_MODEL = {nakdimon.MAIN_MODEL}", flush=True)
    print(f"config attrs = {[a for a in dir(nakdimon.config) if not a.startswith('__')][:20]}", flush=True)

    # 3. Try calling diacritize
    test = "בשנת 1948 השלים אפרים קישון את לימודיו"
    print(f"\n=== Smoke test ===", flush=True)
    print(f"Input: {test}", flush=True)

    # Try the most likely calling convention based on source inspection
    try:
        result = diac_fn(test)
        print(f"Result (text only): {result}", flush=True)
        if isinstance(result, str) and len(result) > 0:
            return _eval_full(diac_fn, test, result)
    except Exception as e:
        print(f"text only failed: {e}", flush=True)

    # Try with model param
    try:
        result = diac_fn(nakdimon.MAIN_MODEL, test)
        print(f"Result (model, text): {result}", flush=True)
    except Exception as e:
        print(f"(model, text) failed: {e}", flush=True)

    # Try importing the actual prediction function
    try:
        from nakdimon.predictor import predict
        print(f"\nFound nakdimon.predictor.predict!", flush=True)
        result = predict(test)
        print(f"predict result: {result}", flush=True)
    except Exception as e:
        print(f"predictor.predict failed: {e}", flush=True)

    return {"status": "inspection complete"}


def _eval_full(diac_fn, smoke_input, smoke_output):
    """Full evaluation if smoke test succeeds."""
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

    examples = examples[:500]
    total_wrong = 0
    total_positions = 0

    for i, (src, gold) in enumerate(examples):
        try:
            pred = diac_fn(src)
            if not isinstance(pred, str):
                pred = src
        except Exception:
            pred = src

        der, n = seq2seq_der(pred, gold)
        total_wrong += int(der * n)
        total_positions += n

        if i < 3:
            print(f"\n  input: {src[:60]}", flush=True)
            print(f"  pred:  {pred[:60]}", flush=True)
            print(f"  gold:  {gold[:60]}", flush=True)

    der = total_wrong / max(1, total_positions)
    print(f"\n=== Nakdimon DER: {der:.4f} ({len(examples)} examples) ===", flush=True)
    return {"model": "Nakdimon", "der": der, "n_examples": len(examples)}


@app.local_entrypoint()
def main():
    result = inspect_nakdimon.remote()
    print(json.dumps(result, indent=2, default=str))

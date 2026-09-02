"""s46 on the Dicta diacritization test corpora (2026-09-01).

The modern-text surface TODO.training-work/07: three public-domain
test sets (Modern/HebrewWiki, Poetry, Rabbinic) from Dicta's ACL 2020
demo, evaluated with the same greedy seq2seq DER harness that produced
s46's Nakdimon number (16.44 greedy). Resumable per example, per
corpus. Results land in dicta_eval.json beside the checkpoint.

Usage:
    modal run --detach eval_hebrew_dicta.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

RUN = "rababa_hebrew/run-s46-phonikud-plus"
_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")

CORPORA = {
    "modern_wiki": "dicta-test/ModernTestCorpus-HebrewWiki1.txt",
    "poetry": "dicta-test/PoetryTestCorpus-Poetry1.txt",
    "rabbinic": "dicta-test/RabbinicTestCorpus-BetYosef1.txt",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy>=1.26,<3", "tqdm>=4.66")
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-hebrew-dicta", image=image)


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate() -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
    from rababa.evaluate import seq2seq_der

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = str(Path("/checkpoints") / RUN / "run-002-gold-ft" / "best")
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to("cuda")
    tokenizer = ByT5Tokenizer.from_pretrained(ckpt)
    model.eval()

    results = {}
    for name, rel in CORPORA.items():
        path = Path("/datasets") / rel
        examples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            undiacritized = "".join(c for c in line if c not in _NIKUD_MARKS).strip()
            if 2 <= len(undiacritized) <= 512:
                examples.append((undiacritized, line))
        print(f"[{name}] {len(examples)} examples", flush=True)

        prog = Path("/checkpoints") / RUN / f"dicta_progress_{name}.jsonl"
        saved: dict[int, tuple[float, int]] = {}
        if prog.exists():
            for l in prog.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    row = json.loads(l)
                    saved[row["i"]] = (row["der"], row["n"])

        missing = [i for i in range(len(examples)) if i not in saved]
        n_new = 0
        with torch.no_grad(), prog.open("a", encoding="utf-8") as out:
            for bi in range(0, len(missing), 16):
                idxs = missing[bi : bi + 16]
                batch = [examples[i][0] for i in idxs]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to("cuda")
                gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
                preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for i, pred in zip(idxs, preds):
                    der, n = seq2seq_der(pred.strip(), examples[i][1])
                    saved[i] = (der, n)
                    out.write(json.dumps({"i": i, "der": der, "n": n}) + "\n")
                n_new += len(idxs)
                if n_new % 320 == 0:
                    out.flush()
                    checkpoints_volume.commit()
                    tw = sum(d * n for d, n in saved.values())
                    tc = sum(n for _, n in saved.values())
                    print(f"[{name}] {len(saved)}/{len(examples)} "
                          f"DER={tw/max(1,tc):.4f} (committed)", flush=True)
        checkpoints_volume.commit()

        total_wrong = sum(d * n for d, n in saved.values())
        total_positions = sum(n for _, n in saved.values())
        der = total_wrong / max(1, total_positions)
        results[name] = {
            "der_greedy": round(der, 4), "n": len(saved), "positions": total_positions,
        }
        print(f"=== {name}: s46 greedy DER {der:.4f} ({len(saved)} examples) ===",
              flush=True)

    result = {
        "checkpoint": ckpt,
        "nakdimon_greedy": 0.1644,
        "corpora": results,
        "source": "Dicta ACL 2020 test corpora (public domain)",
    }
    (Path("/checkpoints") / RUN / "dicta_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(evaluate.remote())

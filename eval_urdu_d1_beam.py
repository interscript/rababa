"""Urdu d1 beam-4 re-eval — collect the free decode win before any d2.

d1 verdict was greedy-only (CER 6.40%, word_acc 47.43%). Hebrew s45
showed beam-4 is worth double-digit points on ByT5 diacritizers.
Same test set, same protocol, line-for-line comparable.

Usage:
    modal run --detach eval_urdu_d1_beam.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

urdu_volume = modal.Volume.from_name("urdu-diacrit-datasets", create_if_missing=False)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

CKPT = "/checkpoints/rababa_urdu_byt5/run-001-d1/best"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "editdistance", "tqdm")
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-urdu-d1-beam", image=image)


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/urdu": urdu_volume, "/checkpoints": checkpoints_volume},
)
def evaluate() -> dict:
    import torch
    import editdistance
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    urdu_volume.reload()
    checkpoints_volume.reload()

    test_pairs = []
    for line in Path("/urdu/urdu-diacrit/test.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        src, tgt = row.get("src", ""), row.get("tgt", "")
        if src and tgt:
            test_pairs.append((src, tgt))
    print(f"[eval] {len(test_pairs)} test pairs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to("cuda")
    model.eval()

    def run(beam: int) -> dict:
        preds = []
        with torch.no_grad():
            for i in range(0, len(test_pairs), 64):
                chunk = test_pairs[i : i + 64]
                enc = tokenizer(
                    [s for s, _ in chunk], return_tensors="pt", padding=True, truncation=True, max_length=1024
                ).to("cuda")
                with torch.autocast("cuda", torch.bfloat16):
                    gen = model.generate(**enc, max_new_tokens=1280, num_beams=beam)
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
                if (i // 64) % 20 == 0:
                    print(f"[gen:beam{beam}] {i + len(chunk)}/{len(test_pairs)}", flush=True)
        total_ed = total_len = exact = 0
        for (_, tgt), pred in zip(test_pairs, preds):
            total_ed += editdistance.eval(pred.strip(), tgt)
            total_len += len(tgt)
            exact += int(pred.strip() == tgt)
        return {"cer": total_ed / max(1, total_len), "word_acc": exact / len(test_pairs)}

    results = {"greedy": run(1), "beam4": run(4)}
    results["baseline_d1_greedy"] = {"cer": 0.0640, "word_acc": 0.4743}
    print(json.dumps(results, indent=2), flush=True)
    out = Path("/checkpoints/rababa_urdu_byt5/run-001-d1/eval_beam.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    checkpoints_volume.commit()
    return results


@app.local_entrypoint()
def main():
    evaluate.remote()

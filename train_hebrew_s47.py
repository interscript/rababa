"""Hebrew s47 — morphological aux-task transplant (TODO.public §3).

The r6 template, first cross-language move: Hebrew's 16.58 residual is
plausibly morphology (nikud encodes gender/number/agreement), the same
shape as Arabic's iʿrāb residual that r6's aux-task fixed.

Design (r6-faithful, s45-housed):
- Stream A (plain): hebrew-v4 gold x2 + 200K phonikud weak pairs.
- Stream B (tagged): dictabert-morph-labeled knesset lines from
  /datasets/hebrew-morph; input "TAG: " + segmented src, target =
  per-token tags ("NOUN|Gender=Masc|..."). The '|' prefix-split marker
  tokens are dropped. B upsampled x4. ASCII prefix on a byte model is
  perfectly distinguishable; plain inference is unaffected.
- Init: s46-best if its final_eval beats s45's 0.1658, else s45-best.
- LR 2e-5 (continued-training regime, r6-proven), 1 epoch, eff. batch 64.
Eval: beam-4 DER on the Nakdimon test, identical harness as s45/s46,
per-example resumable. Gate: beat 16.58.

Usage:
    modal run --detach train_hebrew_s47.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

RUN = "rababa_hebrew/run-s47-morph"
S45_RUN = "rababa_hebrew/run-s45-phonikud"
S46_RUN = "rababa_hebrew/run-s46-phonikud-plus"
S45_DER = 0.1658

N_MORPH = 100_000
N_WEAK = 200_000
TAG_UPSAMPLE = 4
GOLD_UPSAMPLE = 2

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-hebrew-s47", image=image)


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _choose_init() -> tuple[str, float]:
    """s46-best if it beat s45, else s45-best (r7-style dynamic choice)."""
    base = Path("/checkpoints")
    s45 = base / S45_RUN
    s45_der = 0.0
    fe = s45 / "final_eval.json"
    if fe.exists():
        s45_der = json.loads(fe.read_text(encoding="utf-8"))["der"]
    else:
        s45_der = S45_DER
    s46_fe = base / S46_RUN / "final_eval.json"
    if s46_fe.exists():
        s46_der = json.loads(s46_fe.read_text(encoding="utf-8"))["der"]
        if s46_der < s45_der:
            ckpt = base / S46_RUN / "run-002-gold-ft" / "best"
            if ckpt.exists():
                return str(ckpt), s46_der
    return str(s45 / "run-002-gold-ft" / "best"), s45_der


@app.function(
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def train() -> dict:
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    checkpoints_volume.reload()
    datasets_volume.reload()

    done_marker = Path("/checkpoints") / RUN / "EVAL_DONE"
    if done_marker.exists():
        return {"run": RUN, "status": "already-done"}

    init, init_der = _choose_init()
    print(f"[init] {init} (der {init_der:.4f})", flush=True)

    gold = _load_jsonl(Path("/datasets/hebrew-v4/train.jsonl"))
    weak_all = _load_jsonl(Path("/datasets/hebrew-phonikud/pairs/train.jsonl"))
    rng = random.Random(42)
    weak = rng.sample(weak_all, min(N_WEAK, len(weak_all)))
    print(f"[data] gold={len(gold)} weak={len(weak)}", flush=True)

    morph_path = Path("/datasets/hebrew-morph/train.jsonl")
    if not morph_path.exists():
        raise FileNotFoundError(f"{morph_path} missing — run label_hebrew_morph.py first")
    morph_rows = _load_jsonl(morph_path)[:N_MORPH]

    tagged_pairs: list[tuple[str, str]] = []
    for r in morph_rows:
        words, tags = [], []
        for w, t in zip(r["src"].split(), r["tags"]):
            if w == "|":
                continue
            words.append(w)
            tags.append(t)
        if 3 <= len(words) <= 64:
            tagged_pairs.append(("TAG: " + " ".join(words), " ".join(tags)))
    print(f"[data] morph lines={len(morph_rows)} tagged units={len(tagged_pairs)}", flush=True)

    pairs: list[tuple[str, str]] = []
    pairs.extend((r["src"], r["tgt"]) for r in gold * GOLD_UPSAMPLE)
    pairs.extend((r["src"], r["tgt"]) for r in weak)
    pairs.extend(tagged_pairs * TAG_UPSAMPLE)
    rng.shuffle(pairs)
    print(f"[data] total={len(pairs)} "
          f"(gold x{GOLD_UPSAMPLE}={len(gold)*GOLD_UPSAMPLE} weak={len(weak)} "
          f"tagged x{TAG_UPSAMPLE}={len(tagged_pairs)*TAG_UPSAMPLE})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSeq2SeqLM.from_pretrained(init)

    class PairDataset(Dataset):
        def __init__(self, rows: list[tuple[str, str]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            src, tgt = self.rows[idx]
            inputs = tokenizer(src, truncation=True, max_length=512)
            labels = tokenizer(tgt, truncation=True, max_length=512)
            inputs["labels"] = labels["input_ids"]
            return inputs

    val_rows = [(r["src"], r["tgt"]) for r in _load_jsonl(Path("/datasets/hebrew-v4/val.jsonl"))[:200]]

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                checkpoints_volume.commit()
                print(f"[volume] committed at step {state.global_step}", flush=True)
            except Exception as e:
                print(f"[volume] commit failed at step {state.global_step}: {e}", flush=True)

    args = Seq2SeqTrainingArguments(
        output_dir="/checkpoints/" + RUN,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=PairDataset(pairs),
        eval_dataset=PairDataset(val_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[VolumeCommitCallback()],
    )
    import glob

    latest = sorted(glob.glob(f"/checkpoints/{RUN}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = latest[-1] if latest else None
    print(f"[resume] {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    best = Path("/checkpoints") / RUN / "best"
    best.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    checkpoints_volume.commit()

    result = evaluate.remote()
    return {"run": RUN, "init_der": init_der, **result}


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate() -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = str(Path("/checkpoints") / RUN / "best")
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to("cuda")
    tokenizer = ByT5Tokenizer.from_pretrained(ckpt)
    model.eval()

    test_path = Path(_find_nakdimon_root()) / "test.txt"
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        undiacritized = "".join(c for c in line if c not in _NIKUD_MARKS).strip()
        if 2 <= len(undiacritized) <= 512:
            examples.append((undiacritized, line))
    print(f"[eval] {len(examples)} examples", flush=True)

    prog = Path("/checkpoints") / RUN / "eval_progress.jsonl"
    saved: dict[int, tuple[float, int]] = {}
    if prog.exists():
        for l in prog.read_text(encoding="utf-8").splitlines():
            if l.strip():
                row = json.loads(l)
                saved[row["i"]] = (row["der"], row["n"])
        print(f"[eval] resuming with {len(saved)} done", flush=True)

    missing = [i for i in range(len(examples)) if i not in saved]
    n_new = 0
    with torch.no_grad(), prog.open("a", encoding="utf-8") as out:
        for bi in range(0, len(missing), 8):
            idxs = missing[bi : bi + 8]
            batch = [examples[i][0] for i in idxs]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, pred in zip(idxs, preds):
                der, n = seq2seq_der(pred.strip(), examples[i][1])
                saved[i] = (der, n)
                out.write(json.dumps({"i": i, "der": der, "n": n}) + "\n")
            n_new += len(idxs)
            if n_new % 160 == 0:
                out.flush()
                checkpoints_volume.commit()
                total = sum(d * n for d, n in saved.values())
                cnt = sum(n for _, n in saved.values())
                print(f"[eval] {len(saved)}/{len(examples)} DER={total/max(1,cnt):.4f} (committed)", flush=True)
    checkpoints_volume.commit()

    total_wrong = sum(d * n for d, n in saved.values())
    total_positions = sum(n for _, n in saved.values())
    der = total_wrong / max(1, total_positions)
    print(f"=== s47 morph aux DER (beam-4): {der:.4f} ({len(saved)} examples) ===", flush=True)
    result = {"der": der, "n": len(saved), "checkpoint": ckpt,
              "baseline_s45": S45_DER, "init_der": _choose_init()[1]}
    (Path("/checkpoints") / RUN / "final_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (Path("/checkpoints") / RUN / "EVAL_DONE").touch()
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    train.remote()

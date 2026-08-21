"""Hebrew s46 — s45 curriculum with a diversified weak stage.

s45 (16.58% DER, beam-4, Nakdimon test) proved weak-pretrain + gold-FT
using knesset alone (1.5M lines, parliamentary domain). s46 repeats the
recipe verbatim with one change: the weak stage adds ALL of hewiki
Dicta-labeled (label_hewiki_full.py → /datasets/hebrew-hewiki-dicta,
~70K kept lines, encyclopedic domain, decontaminated) on top of the
same s45 knesset pairs. Stage 2 and eval are byte-identical to s45, so
the comparison isolates the weak-diversity variable.

Gate: must beat 16.58 to replace s45; otherwise record flat and keep
s45 as canonical.

Usage:
    modal run --detach train_hebrew_s46.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

RUN = "rababa_hebrew/run-s46-phonikud-plus"
KNESSET_PAIRS = Path("/datasets/hebrew-phonikud/pairs")
HEWIKI_DIR = Path("/datasets/hebrew-hewiki-dicta")
S46_PREP = Path("/datasets/hebrew-hewiki-dicta/s46_pairs.jsonl")

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

app = modal.App("rababa-hebrew-s46", image=image)


@app.function(cpu=8, timeout=1 * 60 * 60, volumes={"/datasets": datasets_volume})
def prep() -> dict:
    datasets_volume.reload()

    if not (HEWIKI_DIR / "DONE").exists():
        return {"status": "waiting-for-label_hewiki_full"}
    if S46_PREP.exists():
        return {"status": "already-done"}

    rows = []
    for line in (HEWIKI_DIR / "train.txt").read_text(encoding="utf-8").splitlines():
        target = line.strip()
        if not target:
            continue
        src = "".join(c for c in target if c not in _NIKUD_MARKS).strip()
        if not src:
            continue
        rows.append({"src": src, "tgt": target})
    random.Random(42).shuffle(rows)
    S46_PREP.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    datasets_volume.commit()
    print(f"[prep] hewiki rows={len(rows)}", flush=True)
    return {"hewiki_rows": len(rows)}


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _mk_dataset_class(tokenizer):
    from torch.utils.data import Dataset

    class JsonlDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            r = self.rows[idx]
            inputs = tokenizer(r["src"], truncation=True, max_length=512)
            labels = tokenizer(r["tgt"], truncation=True, max_length=512)
            inputs["labels"] = labels["input_ids"]
            return inputs

    return JsonlDataset


@app.function(
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def stage1() -> dict:
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

    out_dir = Path("/checkpoints") / RUN / "run-001-pretrain"
    done = out_dir / "DONE"
    if done.exists():
        return {"status": "already-done", "ckpt": str(out_dir / "best")}
    if not S46_PREP.exists():
        return {"status": "run-prep-first"}

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")
    JsonlDataset = _mk_dataset_class(tokenizer)
    train_rows = _load_jsonl(KNESSET_PAIRS / "train.jsonl") + _load_jsonl(S46_PREP)
    val_rows = _load_jsonl(KNESSET_PAIRS / "val.jsonl")
    random.Random(43).shuffle(train_rows)
    print(f"[s1] train={len(train_rows)} (knesset + hewiki) val={len(val_rows)}", flush=True)

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                checkpoints_volume.commit()
                print(f"[volume] committed at step {state.global_step}", flush=True)
            except Exception as e:
                print(f"[volume] commit failed: {e}", flush=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        bf16=True,
        learning_rate=3e-4,
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="steps",
        save_steps=2000,
        eval_strategy="steps",
        eval_steps=2000,
        save_total_limit=1,
        logging_steps=200,
        report_to=[],
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=JsonlDataset(train_rows),
        eval_dataset=JsonlDataset(val_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[VolumeCommitCallback()],
    )

    import glob

    latest = sorted(glob.glob(f"{out_dir}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    trainer.train(resume_from_checkpoint=latest[-1] if latest else None)

    best = out_dir / "best"
    best.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    done.touch()
    checkpoints_volume.commit()
    return {"ckpt": str(best)}


@app.function(
    gpu="A100",
    timeout=8 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def stage2() -> dict:
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

    init = Path("/checkpoints") / RUN / "run-001-pretrain" / "best"
    out_dir = Path("/checkpoints") / RUN / "run-002-gold-ft"
    done = out_dir / "DONE"
    if done.exists():
        return {"status": "already-done", "ckpt": str(out_dir / "best")}

    tokenizer = AutoTokenizer.from_pretrained(str(init))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(init))
    JsonlDataset = _mk_dataset_class(tokenizer)
    train_rows = _load_jsonl(Path("/datasets/hebrew-v4/train.jsonl"))
    val_rows = _load_jsonl(Path("/datasets/hebrew-v4/val.jsonl"))
    print(f"[s2] hebrew-v4 train={len(train_rows)} val={len(val_rows)}", flush=True)

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                checkpoints_volume.commit()
            except Exception as e:
                print(f"[volume] commit failed: {e}", flush=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        bf16=True,
        learning_rate=3e-4,
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to=[],
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=JsonlDataset(train_rows),
        eval_dataset=JsonlDataset(val_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[VolumeCommitCallback()],
    )

    import glob

    latest = sorted(glob.glob(f"{out_dir}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    trainer.train(resume_from_checkpoint=latest[-1] if latest else None)

    best = out_dir / "best"
    best.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    done.touch()
    checkpoints_volume.commit()
    return {"ckpt": str(best)}


@app.function(
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate() -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
    from rababa.evaluate import seq2seq_der
    from rababa.datasets import _find_nakdimon_root

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = str(Path("/checkpoints") / RUN / "run-002-gold-ft" / "best")
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

    # resumable beam-4 generation: per-example predictions on the volume
    prog = Path("/checkpoints") / RUN / "eval_progress.jsonl"
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[eval] resuming with {len(saved)} saved examples", flush=True)

    missing = [i for i in range(len(examples)) if i not in saved]
    n_new = 0
    with torch.no_grad(), prog.open("a", encoding="utf-8") as prog_out:
        for bi in range(0, len(missing), 8):
            idxs = missing[bi : bi + 8]
            batch = [examples[i] for i in idxs]
            enc = tokenizer([s for s, _ in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to("cuda")
            gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, pred in zip(idxs, preds):
                prog_out.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
                saved[i] = pred
            n_new += len(idxs)
            if n_new % 160 == 0:
                prog_out.flush()
                checkpoints_volume.commit()
                print(f"  [{len(saved)}/{len(examples)}] (committed)", flush=True)
    checkpoints_volume.commit()

    total_wrong = total_positions = total_n = 0
    for i, (_, gold) in enumerate(examples):
        der, n = seq2seq_der(saved[i], gold)
        total_wrong += int(der * n)
        total_positions += n
        total_n += 1
        if i % 240 == 0 and i > 0:
            print(f"  [{i}/{len(examples)}] DER={total_wrong / max(1, total_positions):.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    print(f"=== s46 diversified-weak DER (beam-4): {der:.4f} ({total_n} examples) ===", flush=True)
    result = {"der": der, "n": total_n, "checkpoint": ckpt,
              "baseline_s45": 0.1658, "baseline_s43": 0.1746}
    (Path("/checkpoints") / RUN / "final_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(prep.remote(), indent=2))
    print(json.dumps(stage1.remote(), indent=2))
    print(json.dumps(stage2.remote(), indent=2))
    print(json.dumps(evaluate.remote(), indent=2))

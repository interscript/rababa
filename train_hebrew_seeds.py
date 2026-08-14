"""Train additional Hebrew ByT5 seeds for multi-seed ensemble.

Uses the v4 corpus (50K pairs, teamim-preserving format).
Each seed trains from scratch with a different random seed.

Usage:
    modal run --detach train_hebrew_seeds.py::train --seed 43
    modal run --detach train_hebrew_seeds.py::train --seed 44
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
        "transformers>=4.40,<5",
        "sentencepiece",
        "protobuf",
        "accelerate>=1.1.0",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/rababa/src", copy=True)
    .add_local_dir("data", "/opt/rababa/data", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTHONPATH": "/opt/rababa/src"})
)

app = modal.App(name=f"{APP_NAME}-hebrew-seeds", image=image)


@app.function(
    gpu="A100",
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train(seed: int = 43) -> dict:
    """Train ByT5-base on Hebrew v4 corpus with given seed."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from torch.utils.data import Dataset

    datasets_volume.reload()
    checkpoints_volume.reload()

    data_root = Path("/datasets/hebrew-v4")
    train_path = data_root / "train.jsonl"
    val_path = data_root / "val.jsonl"

    if not train_path.is_file():
        return {"error": "No training data at /datasets/hebrew-v4. Run build_combined_corpus first."}

    print(f"[seed-{seed}] loading ByT5-base...", flush=True)
    model_name = "google/byt5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

    class JsonlDataset(Dataset):
        def __init__(self, path, tok, max_len=512):
            self.examples = []
            for ln in Path(path).read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                s = (r.get("src") or "").strip()
                t = (r.get("tgt") or "").strip()
                if s and t:
                    if len(s.encode("utf-8")) > max_len or len(t.encode("utf-8")) > max_len:
                        continue
                    self.examples.append((s, t))
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            s, t = self.examples[idx]
            mi = self.tok(s, truncation=True, max_length=self.max_len)
            lab = self.tok(t, truncation=True, max_length=self.max_len)
            mi["labels"] = lab["input_ids"]
            return mi

    train_ds = JsonlDataset(str(train_path), tokenizer)
    val_ds = JsonlDataset(str(val_path), tokenizer)
    print(f"[seed-{seed}] train={len(train_ds)}, val={len(val_ds)}", flush=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    ckpt_root = Path(f"/checkpoints/rababa_hebrew_byt5_s{seed}/run-001")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_root),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=seed,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        predict_with_generate=False,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    best_path = ckpt_root / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    checkpoints_volume.commit()
    return {"best": str(best_path), "n_train": len(train_ds), "seed": seed}


@app.local_entrypoint()
def main(seed: int = 43):
    result = train.remote(seed=seed)
    print(json.dumps(result, indent=2, default=str))

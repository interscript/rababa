"""Urdu d2 — second epoch at low LR from d1 (beam-4 was flat, TODO 06 rule).

Diagnosis: the shipped Urdu diacritizer (urdu_diacrit/run-001, 14.77%
CER) is a custom char encoder trained on cross-lingually machine-
labeled data. Arabic — same script family, same task shape — reaches
2.68 DER on ByT5-base. This run swaps in the proven architecture with
an Arabic-teacher init.

Data (volume urdu-diacrit-datasets):
- urdu-diacrit/train.jsonl: 573,130 {"src","tgt"} Urdu pairs
  (machine-labeled via our Arabic model + G2P back-projection — WEAK
  labels; no gold Urdu corpus exists in our stack, so this is a
  comparative eval against the 14.77 baseline, not an absolute one).
  NOTE: urdu-diacritized/*.txt is diacritized ARABIC (fiqh text),
  NOT Urdu — deliberately unused here.
- Dedup by src (G2P-derived corpora are repetitive), length caps.

Init: rababa_arabic_byt5/run-005-context/best (ByT5-base) — shared
abjad prior instead of a cold start.

Eval: greedy CER via editdistance on urdu-diacrit/test.jsonl plus
word-level exact match, identical protocol to the 14.77 number.

Usage:
    modal run --detach train_urdu_d1.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

urdu_volume = modal.Volume.from_name("urdu-diacrit-datasets", create_if_missing=False)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_urdu_byt5/run-002-d2"
INIT_RUN = "rababa_urdu_byt5/run-001-d1"
N_VAL = 2_000

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "editdistance",
        "tqdm",
    )
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-urdu-d2", image=image)


def load_pairs(path: str) -> list[tuple[str, str]]:
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src, tgt = row.get("src", ""), row.get("tgt", "")
            if not src or not tgt:
                continue
            if src in seen:
                continue
            if len(src.encode("utf-8")) > 900 or len(tgt.encode("utf-8")) > 1200:
                continue
            if DIACRITICS_RE.search(src):
                continue  # input side must be undiacritized
            if DIACRITICS_RE.sub("", tgt).replace(" ", "") != src.replace(" ", ""):
                continue  # misaligned pair: letters must match after stripping diacritics
            seen.add(src)
            pairs.append((src, tgt))
    return pairs


@app.function(
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/urdu": urdu_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    urdu_volume.reload()
    checkpoints_volume.reload()

    done_marker = Path("/checkpoints") / RUN / "EVAL_DONE"
    if done_marker.exists():
        return {"run": RUN, "status": "already-done"}

    print("[data] loading urdu-diacrit pairs...", flush=True)
    train_pairs = load_pairs("/urdu/urdu-diacrit/train.jsonl")
    val_pairs = load_pairs("/urdu/urdu-diacrit/val.jsonl")[:N_VAL]
    test_pairs = load_pairs("/urdu/urdu-diacrit/test.jsonl")
    random.Random(42).shuffle(train_pairs)
    print(f"[data] train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}", flush=True)

    init = str(Path("/checkpoints") / INIT_RUN / "best")
    print(f"[init] {init}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSeq2SeqLM.from_pretrained(init)

    class PairDataset(Dataset):
        def __init__(self, rows: list[tuple[str, str]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            src, tgt = self.rows[idx]
            inputs = tokenizer(src, truncation=True, max_length=1024)
            labels = tokenizer(tgt, truncation=True, max_length=1280)
            inputs["labels"] = labels["input_ids"]
            return inputs

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
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        bf16=True,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        predict_with_generate=False,
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=PairDataset(train_pairs),
        eval_dataset=PairDataset(val_pairs[:500]),
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

    # ---- greedy CER + word accuracy on the full test split ----
    import editdistance

    device = next(trainer.model.parameters()).device
    trainer.model.eval()

    preds: list[str] = []
    batch = 64
    with torch.no_grad():
        for i in range(0, len(test_pairs), batch):
            chunk = test_pairs[i : i + batch]
            enc = tokenizer(
                [s for s, _ in chunk], return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = trainer.model.generate(**enc, max_new_tokens=1280, num_beams=1)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            if (i // batch) % 20 == 0:
                print(f"[gen] {i + len(chunk)}/{len(test_pairs)}", flush=True)

    total_ed = 0
    total_len = 0
    exact = 0
    for (src, tgt), pred in zip(test_pairs, preds):
        total_ed += editdistance.eval(pred, tgt)
        total_len += len(tgt)
        if pred.strip() == tgt.strip():
            exact += 1
    cer = total_ed / max(1, total_len)
    word_acc = exact / len(test_pairs)
    print(f"[eval] CER={cer:.4f} word_acc={word_acc:.4f} n={len(test_pairs)}", flush=True)

    (Path("/checkpoints") / RUN / "eval.txt").write_text(
        f"CER={cer:.4f}\nword_acc={word_acc:.4f}\nn={len(test_pairs)}\n", encoding="utf-8")
    checkpoints_volume.commit()

    done_marker.touch()
    checkpoints_volume.commit()
    return {"run": RUN, "cer": cer, "word_acc": word_acc}


@app.local_entrypoint()
def main():
    train.remote()

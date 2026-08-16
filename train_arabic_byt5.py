"""Arabic ByT5-base paragraph-level diacritization — beat Claude on SadeedDiac-25.

Why: the char-encoder scores 3.25% DER (CE) on SadeedDiac-25 — ahead of
Sadeed/GPT-4/Gemini, behind Claude-3.7 (1.39). Its two structural limits:
180-char chunked context (case endings need whole-sentence syntax) and
argmax decoding. ByT5-base seq2seq reads whole paragraphs and decodes
with beam search — the same change that won Hebrew by 18 points over
DictaBERT.

Training: 800K-line subsample of arabic-combined/train.txt (seed 42),
src = haraqat-stripped, tgt = original, 1 epoch, cosine, batch 16×2 accum.
Eval: full SadeedDiac-25 (1,200 paragraphs) with Misraj's own evaluator,
beam 4, plus beam 1.

Usage:
    modal run --detach train_arabic_byt5.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_byt5/run-002-full-2ep"
DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
N_TRAIN = 9_999_999
N_VAL = 2_000
MAX_BYTES = 640

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "editdistance",
        "pyarabic",
        "prettytable",
        "pandas",
        "tqdm",
        "pyarrow",
    )
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-byt5", image=image)


@app.function(
    gpu="A100",
    timeout=11 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
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

    datasets_volume.reload()

    done_marker = Path("/checkpoints") / RUN / "EVAL_DONE"
    if done_marker.exists():
        print("[done] already trained+evaluated, nothing to do", flush=True)
        return {"run": RUN, "status": "already-done"}

    print("[data] loading corpus...", flush=True)
    lines = [
        l.strip()
        for l in Path("/datasets/arabic-combined/train.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"[data] {len(lines)} lines", flush=True)

    rng = random.Random(42)
    rng.shuffle(lines)

    def make_pair(line: str) -> tuple[str, str] | None:
        src = DIACRITICS_RE.sub("", line)
        if not src:
            return None
        if len(src.encode("utf-8")) > MAX_BYTES or len(line.encode("utf-8")) > MAX_BYTES:
            return None
        return src, line

    val_pairs = [p for p in (make_pair(l) for l in lines[:N_VAL]) if p]
    train_pairs = [p for p in (make_pair(l) for l in lines[N_VAL : N_VAL + N_TRAIN]) if p]
    print(f"[data] train={len(train_pairs)} val={len(val_pairs)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")

    class LineDataset(Dataset):
        def __init__(self, pairs: list[tuple[str, str]]) -> None:
            self.pairs = pairs

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> dict:
            src, tgt = self.pairs[idx]
            inputs = tokenizer(src, truncation=True, max_length=MAX_BYTES)
            labels = tokenizer(tgt, truncation=True, max_length=MAX_BYTES)
            inputs["labels"] = labels["input_ids"]
            return inputs

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    args = Seq2SeqTrainingArguments(
        output_dir="/checkpoints/" + RUN,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        bf16=True,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=300,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="steps",
        save_steps=4000,
        eval_strategy="epoch",
        save_total_limit=1,
        logging_steps=200,
        report_to=[],
        predict_with_generate=False,
        dataloader_num_workers=4,
    )
    # Preempted containers lose their uncommitted volume overlay; commit at every
    # checkpoint save so an eviction costs at most save_steps of progress.
    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                checkpoints_volume.commit()
                print(f"[volume] committed at step {state.global_step}", flush=True)
            except Exception as e:
                print(f"[volume] commit failed at step {state.global_step}: {e}", flush=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=LineDataset(train_pairs),
        eval_dataset=LineDataset(val_pairs),
        data_collator=collator,
        callbacks=[VolumeCommitCallback()],
    )
    import glob
    latest = sorted(glob.glob("/checkpoints/rababa_arabic_byt5/run-002-full-2ep/checkpoint-*"))
    resume = latest[-1] if latest else None
    print(f"[resume] {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    best = Path("/checkpoints") / RUN / "best"
    best.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    checkpoints_volume.commit()

    # ---- SadeedDiac-25 with Misraj's evaluator ----
    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    device = next(trainer.model.parameters()).device
    trainer.model.eval()

    preds_by_beam: dict[int, list[str]] = {}
    for beam in (4, 1):
        preds: list[str] = []
        with torch.no_grad():
            for i in range(0, len(inputs), 16):
                batch = inputs[i : i + 16]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
                ).to(device)
                gen = trainer.model.generate(**enc, max_new_tokens=1024, num_beams=beam)
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
                if (i // 16) % 20 == 0:
                    print(f"[gen beam={beam}] {i + len(batch)}/{len(inputs)}", flush=True)
        preds_by_beam[beam] = preds

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    results: dict = {"run": RUN}
    for beam in (4, 1):
        csv_path = Path(f"/tmp/sadeed_byt5_beam{beam}.csv")
        pd.DataFrame({"gt": outputs, "pred": preds_by_beam[beam]}).to_csv(csv_path, index=False, header=False)
        print(f"\n===== ByT5-base beam={beam} (their default protocol) =====", flush=True)
        E.report_errors_on_csv_file(
            str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
            gt_missing_diacritic_is_error=False,
        )
        Path(f"/checkpoints/{RUN}/sadeed_preds_beam{beam}.csv").write_text(csv_path.read_text())
    done_marker.touch()
    checkpoints_volume.commit()
    return results


@app.local_entrypoint()
def main():
    train.remote()

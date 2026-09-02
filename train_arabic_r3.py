"""Arabic r3 — domain-adaptation SFT of ByT5 r2 on decontaminated Misraj corpus.

Why: r2 scores 2.94/1.83 DER on SadeedDiac-25 with 1.40% on in-domain dev
— the residual is domain gap (the benchmark is 50% Classical Arabic), not
optimization: 67% of r2's errors are word-internal vowel confusions, only
33% word-final iʿrāb. Misraj's public corpus is the benchmark's source
distribution, but 122 benchmark paragraphs appear verbatim in it plus ~1k
near-duplicate lines, so we train only on the decontaminated copy
(sadeed-decontam/train.txt: 60-char window, stride-1 both sides, stricter
than the 13-gram field standard) mixed with an arabic-combined replay
slice to protect MSA.

Init: r2 best (rababa_arabic_byt5/run-002-full-2ep/best). 1M decontam +
150k replay, 1 epoch, LR 3e-5 cosine. Per-save volume commits; EVAL_DONE
marker; final SadeedDiac-25 with Misraj's evaluator (beam 4 + 1).

Usage:
    modal run --detach train_arabic_r3.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_byt5/run-003-domain"
SFT_RUN = "rababa_arabic_byt5/run-002-full-2ep"
N_MISRAJ = 1_000_000
N_REPLAY = 150_000
N_VAL = 2_000
MAX_BYTES = 640

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
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

app = modal.App("rababa-arabic-r3", image=image)


@app.function(
    gpu="A100",
    timeout=11 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
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

    datasets_volume.reload()
    checkpoints_volume.reload()

    done_marker = Path("/checkpoints") / RUN / "EVAL_DONE"
    if done_marker.exists():
        return {"run": RUN, "status": "already-done"}

    def make_pair(line: str) -> tuple[str, str] | None:
        src = DIACRITICS_RE.sub("", line)
        if not src:
            return None
        if len(src.encode("utf-8")) > MAX_BYTES or len(line.encode("utf-8")) > MAX_BYTES:
            return None
        return src, line

    print("[data] loading decontaminated Misraj corpus...", flush=True)
    misraj = [l.strip() for l in Path("/datasets/sadeed-decontam/train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(45).shuffle(misraj)
    misraj_pairs = [p for p in (make_pair(l) for l in misraj[: int(N_MISRAJ * 1.1)]) if p][:N_MISRAJ]

    combined = [l.strip() for l in Path("/datasets/arabic-combined/train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(44).shuffle(combined)
    replay_pairs = [p for p in (make_pair(l) for l in combined[N_VAL : N_VAL + int(N_REPLAY * 1.2)]) if p][:N_REPLAY]
    val_pairs = [p for p in (make_pair(l) for l in combined[:N_VAL]) if p][:200]

    pairs = misraj_pairs + replay_pairs
    random.Random(42).shuffle(pairs)
    print(f"[data] misraj={len(misraj_pairs)} replay={len(replay_pairs)} total={len(pairs)} val={len(val_pairs)}", flush=True)

    init = str(Path("/checkpoints") / SFT_RUN / "best")
    print(f"[init] {init}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSeq2SeqLM.from_pretrained(init)

    class LineDataset(Dataset):
        def __init__(self, rows: list[tuple[str, str]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            src, tgt = self.rows[idx]
            inputs = tokenizer(src, truncation=True, max_length=MAX_BYTES)
            labels = tokenizer(tgt, truncation=True, max_length=MAX_BYTES)
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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        bf16=True,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=300,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="steps",
        save_steps=3000,
        eval_strategy="epoch",
        save_total_limit=1,
        logging_steps=200,
        report_to=[],
        predict_with_generate=False,
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=LineDataset(pairs),
        eval_dataset=LineDataset(val_pairs),
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

    import pandas as pd
    import pyarrow.parquet as pq

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    device = next(trainer.model.parameters()).device
    trainer.model.eval()

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    results: dict = {"run": RUN}
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
        csv_path = Path(f"/tmp/sadeed_r3_beam{beam}.csv")
        pd.DataFrame({"gt": outputs, "pred": preds}).to_csv(csv_path, index=False, header=False)
        print(f"\n===== r3 beam={beam} (their default protocol) =====", flush=True)
        E.report_errors_on_csv_file(
            str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
            gt_missing_diacritic_is_error=False,
        )
        (Path("/checkpoints") / RUN / f"sadeed_preds_beam{beam}.csv").write_text(
            csv_path.read_text(), encoding="utf-8"
        )
        checkpoints_volume.commit()

    done_marker.touch()
    checkpoints_volume.commit()
    return results


@app.local_entrypoint()
def main():
    train.remote()

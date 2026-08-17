"""Arabic r5 — paragraph-context training (close the context gap to LLMs).

Diagnosis chain: our residual errors are phonotactically LEGAL alternate
readings (98.7%) — a discrimination/context problem, not phonology.
GLM-5.2's edge on this benchmark is whole-paragraph reading; we train on
isolated 640-byte lines and evaluate in 600-byte windows. The
decontaminated Misraj corpus is line-split continuous book text, so
joining k adjacent lines reconstructs true paragraphs with real
inter-sentence context — exactly what case endings (iʿrāb) need.

r5 = r3 best + 1 epoch on joined ~1400-byte paragraph units (domain
corpus) + joined replay. Eval: windowed at 1400 bytes with 2x generation
cap and haraqat projection (zero-skip protocol).

Usage:
    modal run --detach train_arabic_r5.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_byt5/run-005-context"
INIT_RUN = "rababa_arabic_byt5/run-003-domain"
UNIT_BYTES = 1400
N_DOMAIN_UNITS = 500_000
N_REPLAY_UNITS = 100_000
N_VAL = 2_000

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

app = modal.App("rababa-arabic-r5", image=image)


def join_units(paths: list[Path], budget: int, limit: int, seed: int) -> list[str]:
    """Join consecutive lines of continuous book text into ~budget-byte units."""
    units: list[str] = []
    for path in paths:
        cur: list[str] = []
        n = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            c = len(line.encode("utf-8")) + 1
            if cur and n + c > budget:
                units.append(" ".join(cur))
                cur, n = [], 0
                if len(units) >= limit * 2:
                    break
            cur.append(line)
            n += c
        if cur:
            units.append(" ".join(cur))
    random.Random(seed).shuffle(units)
    return units[:limit]


def make_pair(unit: str) -> tuple[str, str] | None:
    src = DIACRITICS_RE.sub("", unit)
    if not src:
        return None
    if len(src.encode("utf-8")) > 1600 or len(unit.encode("utf-8")) > 1600:
        return None
    return src, unit


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

    print("[data] joining domain corpus into paragraph units...", flush=True)
    # Cache joined units on the volume: preemption relaunches re-paid the full
    # join (~10 min of CPU) every time.
    cache = Path("/datasets/r5-units")
    if (cache / "domain.txt").exists() and (cache / "replay.txt").exists():
        domain = [l for l in (cache / "domain.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        replay = [l for l in (cache / "replay.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        random.Random(42).shuffle(domain)
        random.Random(42).shuffle(replay)
        domain = domain[:N_DOMAIN_UNITS]
        replay = replay[:N_REPLAY_UNITS]
        print(f"[data] loaded from cache: domain={len(domain)} replay={len(replay)}", flush=True)
    else:
        domain = join_units([Path("/datasets/sadeed-decontam/train.txt")], UNIT_BYTES, N_DOMAIN_UNITS, seed=46)
        replay = join_units([Path("/datasets/arabic-combined/train.txt")], UNIT_BYTES, N_REPLAY_UNITS, seed=47)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "domain.txt").write_text("\n".join(domain) + "\n", encoding="utf-8")
        (cache / "replay.txt").write_text("\n".join(replay) + "\n", encoding="utf-8")
        datasets_volume.commit()
        print(f"[data] cached joined units: domain={len(domain)} replay={len(replay)}", flush=True)

    domain_pairs = [p for p in (make_pair(u) for u in domain) if p]
    replay_pairs = [p for p in (make_pair(u) for u in replay) if p]

    combined = [l.strip() for l in Path("/datasets/arabic-combined/train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(42).shuffle(combined)
    val_pairs = [p for p in (make_pair(l) for l in combined[:N_VAL]) if p][:200]

    pairs = domain_pairs + replay_pairs
    random.Random(42).shuffle(pairs)
    print(f"[data] domain={len(domain_pairs)} replay={len(replay_pairs)} total={len(pairs)} val={len(val_pairs)}", flush=True)

    init = str(Path("/checkpoints") / INIT_RUN / "best")
    print(f"[init] {init}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSeq2SeqLM.from_pretrained(init)

    class UnitDataset(Dataset):
        def __init__(self, rows: list[tuple[str, str]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            src, tgt = self.rows[idx]
            inputs = tokenizer(src, truncation=True, max_length=1600)
            labels = tokenizer(tgt, truncation=True, max_length=1600)
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        bf16=True,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        predict_with_generate=False,
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=UnitDataset(pairs),
        eval_dataset=UnitDataset(val_pairs),
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

    # ---- windowed zero-skip eval at the training context size ----
    import pandas as pd
    import pyarrow.parquet as pq
    from difflib import SequenceMatcher

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    def split_windows(text: str, budget: int = UNIT_BYTES) -> list[str]:
        if len(text.encode("utf-8")) <= budget:
            return [text]
        words = text.split()
        wins, cur, n = [], [], 0
        for w in words:
            c = len(w.encode("utf-8")) + 1
            if cur and n + c > budget:
                wins.append(" ".join(cur))
                cur, n = [], 0
            cur.append(w)
            n += c
        if cur:
            wins.append(" ".join(cur))
        return wins

    def project_haraqat(pred: str, text: str) -> str:
        pred_haraqat = [""]
        for ch in pred:
            if DIACRITICS_RE.match(ch):
                pred_haraqat[-1] += ch
            else:
                pred_haraqat.append("")
        pred_haraqat = pred_haraqat[1:]
        pred_letters = [c for c in pred if not DIACRITICS_RE.match(c)]
        text_letters = [c for c in text if not DIACRITICS_RE.match(c)]
        sm = SequenceMatcher(None, text_letters, pred_letters, autojunk=False)
        out = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    out.append(text_letters[i1 + k] + pred_haraqat[j1 + k])
            else:
                for k in range(i1, i2):
                    out.append(text_letters[k])
        return "".join(out)

    device = next(trainer.model.parameters()).device
    trainer.model.eval()

    all_windows: list[str] = []
    counts: list[int] = []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(inputs)} paragraphs -> {len(all_windows)} windows", flush=True)

    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(all_windows), 8):
            batch = all_windows[i : i + 8]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1600
            ).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = trainer.model.generate(**enc, max_new_tokens=3200, num_beams=1)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            if (i // 8) % 40 == 0:
                print(f"[gen] {i + len(batch)}/{len(all_windows)}", flush=True)

    k = 0
    paragraphs = []
    for text, c in zip(inputs, counts):
        stitched = " ".join(preds[k : k + c])
        k += c
        paragraphs.append(project_haraqat(stitched, text))

    csv_path = Path("/tmp/sadeed_r5_windowed.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (Path("/checkpoints") / RUN / "sadeed_preds_windowed.csv").write_text(
        csv_path.read_text(), encoding="utf-8")
    checkpoints_volume.commit()

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("\n===== r5 paragraph-context, windowed zero-skip =====", flush=True)
    E.report_errors_on_csv_file(
        str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
        gt_missing_diacritic_is_error=False)

    done_marker.touch()
    checkpoints_volume.commit()
    return {"run": RUN}


@app.local_entrypoint()
def main():
    train.remote()

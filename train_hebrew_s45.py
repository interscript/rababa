"""Hebrew s45 — phonikud curriculum: modern-Hebrew pretrain, then gold FT.

Hypothesis: s43 (17.46% DER on Nakdimon test, beam-4) is data-limited
(~60K gold examples). Phonikud knesset v6 (5.3M machine-labeled lines,
Dicta diacritizer + ~1k manual high-freq corrections) provides a 25x
larger WEAK-supervision corpus for a PRETRAIN stage only. Stage 2
re-runs s43's exact gold recipe on top, so Dicta's systematic errors are
corrected by the gold fine-tune — the weak stage supplies a nikud prior,
not the ceiling (no-teacher-poison rule honored).

Data prep (single-column v6 file: nikud + '|' syllable marks + stress):
- target = line with non-standard marks dropped (keep Hebrew letters,
  ASCII, whitespace, project nikud set); input = target minus nikud.
- filters: length, Hebrew fraction, min-nikud density; exact-input dedupe
  (knesset is formulaic); 40-char window decontam vs Nakdimon test.
Stage 1: ByT5-base, 1 epoch, batch 64, LR 3e-4 (proven here), 1.5M cap.
Stage 2: s43 recipe verbatim (hebrew-v4 jsonl, 3 ep, batch 8, LR 3e-4,
warmup 500) from the stage-1 checkpoint.
Eval: beam-4 DER, identical protocol/harness as the s43 number.

Usage:
    modal run --detach train_hebrew_s45.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

RUN = "rababa_hebrew/run-s45-phonikud"
SRC = Path("/datasets/hebrew-phonikud/knesset_nikud_v6.txt")
PAIRS_DIR = Path("/datasets/hebrew-phonikud/pairs")
STAGE1_N = 1_500_000

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

app = modal.App("rababa-hebrew-s45", image=image)


def _is_hebrew_letter(c: str) -> bool:
    return "א" <= c <= "ת"


def _normalize(line: str) -> tuple[str, str] | None:
    tgt_chars = []
    for c in line:
        if c in _NIKUD_MARKS or _is_hebrew_letter(c) or c.isspace() or (33 <= ord(c) <= 126):
            tgt_chars.append(c)
        # drop: '|', teamim/stress, phonikud custom marks, anything else
    target = "".join(tgt_chars).strip()
    src = "".join(c for c in target if c not in _NIKUD_MARKS).strip()
    if not (10 <= len(src) <= 300):
        return None
    letters = sum(1 for c in src if _is_hebrew_letter(c))
    if letters / max(1, len(src.replace(" ", ""))) < 0.6:
        return None
    nikud = sum(1 for c in target if c in _NIKUD_MARKS)
    if nikud < len(src.replace(" ", "")) / 12:
        return None
    return src, target


@app.function(cpu=8, timeout=3 * 60 * 60, volumes={"/datasets": datasets_volume})
def prep() -> dict:
    datasets_volume.reload()

    marker = PAIRS_DIR / "PREP_DONE"
    if marker.exists():
        return {"status": "already-done"}

    from rababa.datasets import _find_nakdimon_root

    test_path = Path(_find_nakdimon_root()) / "test.txt"
    test_windows: set[str] = set()
    for line in test_path.read_text(encoding="utf-8").splitlines():
        stripped = "".join(c for c in line.strip() if c not in _NIKUD_MARKS)
        for i in range(0, max(1, len(stripped) - 40), 20):
            test_windows.add(stripped[i : i + 40])
    print(f"[decontam] {len(test_windows)} test windows", flush=True)

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAIRS_DIR / "train.jsonl"
    seen: set[str] = set()
    n_kept = n_dup = n_contam = n_filtered = 0
    with open(SRC, encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            if n_kept >= STAGE1_N + 5_000:
                break
            pair = _normalize(line)
            if pair is None:
                n_filtered += 1
                continue
            src, target = pair
            if src in seen:
                n_dup += 1
                continue
            if any(src[i : i + 40] in test_windows for i in range(0, max(1, len(src) - 40), 20)):
                n_contam += 1
                continue
            seen.add(src)
            out.write(json.dumps({"src": src, "tgt": target}, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept % 200_000 == 0:
                print(f"  [prep] kept={n_kept} dup={n_dup} contam={n_contam} filt={n_filtered}", flush=True)

    rows = out_path.read_text(encoding="utf-8").splitlines()
    random.Random(42).shuffle(rows)
    val = rows[:2_000]
    train = rows[2_000 : 2_000 + STAGE1_N]
    out_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    (PAIRS_DIR / "val.jsonl").write_text("\n".join(val) + "\n", encoding="utf-8")

    marker.touch()
    datasets_volume.commit()
    print(f"[prep] final train={len(train)} val={len(val)} dup={n_dup} contam={n_contam} "
          f"filt={n_filtered}", flush=True)
    return {"train": len(train), "val": len(val), "dup": n_dup, "contam": n_contam}


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
    import torch
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

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")
    JsonlDataset = _mk_dataset_class(tokenizer)
    train_rows = _load_jsonl(PAIRS_DIR / "train.jsonl")
    val_rows = _load_jsonl(PAIRS_DIR / "val.jsonl")
    print(f"[s1] train={len(train_rows)} val={len(val_rows)}", flush=True)

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

    total_wrong = total_positions = total_n = 0
    with torch.no_grad():
        for i in range(0, len(examples), 8):
            batch = examples[i : i + 8]
            enc = tokenizer([s for s, _ in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to("cuda")
            gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for pred, (_, gold) in zip(preds, batch):
                der, n = seq2seq_der(pred, gold)
                total_wrong += int(der * n)
                total_positions += n
                total_n += 1
            if i % 240 == 0 and i > 0:
                print(f"  [{i}/{len(examples)}] DER={total_wrong/max(1,total_positions):.4f}", flush=True)

    der = total_wrong / max(1, total_positions)
    print(f"=== s45 phonikud curriculum DER (beam-4): {der:.4f} ({total_n} examples) ===", flush=True)
    result = {"der": der, "n": total_n, "checkpoint": ckpt,
              "baseline_s43": 0.1746}
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

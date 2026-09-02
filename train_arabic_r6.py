"""Arabic r6 — morphological aux-task training (iʿrāb supervision).

Diagnosis: 33% of residual errors are word-final case endings; the
Total-vs-Morph DER gap (2.68 vs 1.60) makes iʿrāb roughly half the
problem. qalsadi-labeled corpus (300k lines, 68.6% exact case/tense +
26.3% coarse POS) now exists on the volume. The RL campaign proved the
residual is knowledge-limited — so we ADD knowledge via an auxiliary
task, not policy sharpening.

Design: two-format multitask on one ByT5.
- Stream A (plain): r5's paragraph units verbatim (cached r5-units).
- Stream B (tagged): morph lines joined into ~1000B units; input gets
  an ASCII "TAG: " prefix (byte model => perfectly distinguishable),
  target = diacritized text + " ||| " + per-word tags. Deterministic:
  inference without the prefix always yields plain diacritization.
B is upsampled x4 to ~25% of the mix. Init from r5 best, A100-80GB,
r5-proven batch 2/accum 15. Eval: windowed zero-skip at 1400B.

Usage:
    modal run --detach train_arabic_r6.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_byt5/run-006-morph"
INIT_RUN = "rababa_arabic_byt5/run-005-context"
UNIT_BYTES = 1400
TAG_UNIT_BYTES = 1000
TAG_UPSAMPLE = 4
N_VAL = 2_000

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "pandas",
        "tqdm",
        "pyarrow",
        "pyarabic",
        "prettytable",
    )
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-r6", image=image)


def join_tagged(rows: list[dict], budget: int) -> list[tuple[str, str, str]]:
    """Join consecutive morph-labeled lines into <=budget-byte tagged units."""
    units: list[tuple[str, str, str]] = []
    cur_src: list[str] = []
    cur_gold: list[str] = []
    cur_tags: list[str] = []
    n = 0
    for r in rows:
        src, gold, tags = r["src"], r["gold"], r["tags"]
        if not src or len(src.split()) != len(tags):
            continue
        c = len(src.encode("utf-8")) + 1
        if cur_src and n + c > budget:
            units.append((" ".join(cur_src), " ".join(cur_gold), " ".join(cur_tags)))
            cur_src, cur_gold, cur_tags, n = [], [], [], 0
        cur_src.append(src)
        cur_gold.append(gold)
        cur_tags.extend(tags)
        n += c
    if cur_src:
        units.append((" ".join(cur_src), " ".join(cur_gold), " ".join(cur_tags)))
    return units


def make_pair(unit: str) -> tuple[str, str] | None:
    src = DIACRITICS_RE.sub("", unit)
    if not src:
        return None
    if len(src.encode("utf-8")) > 1450 or len(unit.encode("utf-8")) > 1450:
        return None
    return src, unit


@app.function(
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
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

    print("[data] loading cached r5 paragraph units...", flush=True)
    cache = Path("/datasets/r5-units")
    domain = [l for l in (cache / "domain.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    replay = [l for l in (cache / "replay.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(42).shuffle(domain)
    random.Random(42).shuffle(replay)

    print("[data] loading morph-labeled lines...", flush=True)
    morph_rows = [
        json.loads(l)
        for l in Path("/datasets/arabic-morph/train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    tagged_units = join_tagged(morph_rows, TAG_UNIT_BYTES)
    print(f"[data] {len(tagged_units)} tagged units from {len(morph_rows)} lines", flush=True)

    pairs: list[tuple[str, str]] = []
    pairs.extend(p for p in (make_pair(u) for u in domain) if p)
    pairs.extend(p for p in (make_pair(u) for u in replay) if p)
    tagged_pairs: list[tuple[str, str]] = []
    for src, gold, tags in tagged_units:
        if len(gold.encode("utf-8")) > TAG_UNIT_BYTES + 450:
            continue
        tagged_pairs.append(("TAG: " + src, gold + " ||| " + tags))
    pairs.extend(tagged_pairs * TAG_UPSAMPLE)
    random.Random(42).shuffle(pairs)
    print(f"[data] plain={len(domain)+len(replay)} tagged={len(tagged_pairs)} (x{TAG_UPSAMPLE}) total={len(pairs)}", flush=True)

    combined = [l.strip() for l in Path("/datasets/arabic-combined/train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(42).shuffle(combined)
    val_pairs = [p for p in (make_pair(l) for l in combined[:N_VAL]) if p][:200]

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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=15,
        per_device_eval_batch_size=1,
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="steps",
        save_steps=300,
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

    # resumable generation: append per-window predictions to the volume
    import json as _json
    prog = Path("/checkpoints") / RUN / "eval_progress.jsonl"
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[gen] resuming with {len(saved)} saved windows", flush=True)

    missing = [i for i in range(len(all_windows)) if i not in saved]
    n_new = 0
    with torch.no_grad(), prog.open("a", encoding="utf-8") as prog_out:
        for bi in range(0, len(missing), 8):
            idxs = missing[bi : bi + 8]
            batch = [all_windows[i] for i in idxs]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1600
            ).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = trainer.model.generate(**enc, max_new_tokens=3200, num_beams=1)
            batch_preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, pred in zip(idxs, batch_preds):
                prog_out.write(_json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
                saved[i] = pred
            n_new += len(idxs)
            if n_new % 160 == 0:
                prog_out.flush()
                checkpoints_volume.commit()
                print(f"[gen] {len(saved)}/{len(all_windows)} (committed)", flush=True)
    checkpoints_volume.commit()
    preds = [saved[i] for i in range(len(all_windows))]

    k = 0
    paragraphs = []
    for text, c in zip(inputs, counts):
        stitched = " ".join(preds[k : k + c])
        k += c
        paragraphs.append(project_haraqat(stitched, text))

    csv_path = Path("/tmp/sadeed_r6_windowed.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (Path("/checkpoints") / RUN / "sadeed_preds_windowed.csv").write_text(
        csv_path.read_text(), encoding="utf-8")
    checkpoints_volume.commit()

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("\n===== r6 morph aux-task, windowed zero-skip =====", flush=True)
    E.report_errors_on_csv_file(
        str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
        gt_missing_diacritic_is_error=False)

    done_marker.touch()
    checkpoints_volume.commit()
    return {"run": RUN}


@app.local_entrypoint()
def main():
    train.remote()

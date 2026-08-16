"""RAFT (rejection-sampling fine-tuning) on the Arabic ByT5 SFT model.

Verifiable-reward RL v1 from the GLM-5.3 playbook (TODO.research/12):
sample K candidates on corpus prompts, score each by letter-aligned DER
against gold haraqat — a deterministic oracle, no hacking surface — and
fine-tune only on samples that strictly beat the greedy baseline.
Model selection happens on the frozen private dev split; SadeedDiac-25
is measured once, at the very end.

Chain: waits for the SFT run's EVAL_DONE marker, then 3 RAFT iterations
(6k prompts x K=4 samples each), then the full SadeedDiac-25 benchmark
with Misraj's evaluator. Per-iter volume commits + markers make any
preemption resume where it left off.

Usage:
    modal run --detach train_arabic_raft.py
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

SFT_RUN = "rababa_arabic_byt5/run-002-full-2ep"
RAFT_RUN = "rababa_arabic_raft/run-001"
N_VAL = 2_000
PROMPT_POOL = 200_000
N_PROMPTS = 6_000
K = 4
TEMP = 0.9
TOP_P = 0.95
MAX_BYTES = 640
ITERS = 3
LR = 3e-5
DEV_N = 500
KEEP_MAX_DER = 0.10
SFT_WAIT_POLLS = 24  # x 5 min = 2h per relaunch

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "pandas",
        "pyarrow",
        "tqdm",
    )
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-raft", image=image)


def letter_haraqat(text: str) -> list[list[str]]:
    seq: list[list[str]] = []
    for ch in text:
        if DIACRITICS_RE.match(ch):
            if seq:
                seq[-1][1] += ch
        else:
            seq.append([ch, ""])
    return seq


def der(pred: str, gold: str) -> float:
    """Letter-aligned haraqat error rate; 1.0 = corrupted letters (reject)."""
    p, g = letter_haraqat(pred), letter_haraqat(gold)
    if len(p) != len(g) or any(a[0] != b[0] for a, b in zip(p, g)):
        return 1.0
    err = tot = 0
    for a, b in zip(p, g):
        if a[0] == " ":
            continue
        if b[1]:
            tot += 1
            err += a[1] != b[1]
        elif a[1]:
            tot += 1
            err += 1
    return err / tot if tot else 0.0


def load_splits() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    lines = [
        l.strip()
        for l in Path("/datasets/arabic-combined/train.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    random.Random(42).shuffle(lines)

    def valid(line: str) -> tuple[str, str] | None:
        src = DIACRITICS_RE.sub("", line)
        if not src:
            return None
        if len(src.encode("utf-8")) > MAX_BYTES or len(line.encode("utf-8")) > MAX_BYTES:
            return None
        return src, line

    dev = [p for p in (valid(l) for l in lines[:N_VAL]) if p][:DEV_N]
    pool: list[tuple[str, str]] = []
    for line in lines[N_VAL:]:
        p = valid(line)
        if p:
            pool.append(p)
        if len(pool) >= PROMPT_POOL:
            break
    random.Random(43).shuffle(pool)
    return dev, pool[:N_PROMPTS]


@app.function(
    gpu="A100",
    timeout=11 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def run() -> dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    raft_dir = Path("/checkpoints") / RAFT_RUN
    done_marker = raft_dir / "EVAL_DONE"
    checkpoints_volume.reload()
    if done_marker.exists():
        return {"status": "already-done"}

    # Wait for the SFT run to finish its own eval before touching anything.
    sft_done = Path("/checkpoints") / SFT_RUN / "EVAL_DONE"
    for _ in range(SFT_WAIT_POLLS):
        if sft_done.exists() and (Path("/checkpoints") / SFT_RUN / "best").is_dir():
            break
        print("[wait] SFT not done yet, sleeping 5min", flush=True)
        time.sleep(300)
        checkpoints_volume.reload()
    else:
        return {"status": "sft-not-ready"}

    sft_best = str(Path("/checkpoints") / SFT_RUN / "best")
    # resume from best-so-far when mid-run preemption lost later iterations
    load_dir = str(raft_dir / "best") if (raft_dir / "best").is_dir() else sft_best
    print(f"[load] {load_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(load_dir).to("cuda")
    device = next(model.parameters()).device

    dev, prompts = load_splits()
    print(f"[data] dev={len(dev)} prompts={len(prompts)}", flush=True)

    def greedy(texts: list[str], batch: int = 32) -> list[str]:
        out: list[str] = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tokenizer(
                    texts[i : i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_BYTES,
                ).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = model.generate(**enc, max_new_tokens=MAX_BYTES, num_beams=1)
                out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out

    def mean_der(pairs: list[tuple[str, str]]) -> float:
        preds = greedy([src for src, _ in pairs])
        ders = [der(p, gold) for p, (_, gold) in zip(preds, pairs)]
        return sum(ders) / len(ders)

    metrics_path = raft_dir / "metrics.jsonl"
    raft_dir.mkdir(parents=True, exist_ok=True)

    best_dev = None
    if (raft_dir / "best").is_dir():
        # resuming after preemption: recompute the SFT baseline only if absent
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                import json

                m = json.loads(line)
                if m.get("best_dev") is not None:
                    best_dev = m["best_dev"]
    if best_dev is None:
        best_dev = mean_der(dev)
        print(f"[dev] SFT baseline DER={best_dev:.4%}", flush=True)

    class PairDataset(Dataset):
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

    for it in range(1, ITERS + 1):
        iter_marker = raft_dir / f"iter{it}.done"
        if iter_marker.exists():
            continue

        srcs = [src for src, _ in prompts]
        golds = [gold for _, gold in prompts]
        print(f"[iter{it}] sampling greedy + {K} candidates on {len(srcs)} prompts", flush=True)

        greedy_preds: list[str] = []
        samples: list[list[str]] = [[] for _ in srcs]
        model.eval()
        with torch.no_grad():
            for i in range(0, len(srcs), 16):
                batch = srcs[i : i + 16]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_BYTES,
                ).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    g = model.generate(**enc, max_new_tokens=MAX_BYTES, num_beams=1)
                    s = model.generate(
                        **enc, max_new_tokens=MAX_BYTES, num_beams=1,
                        do_sample=True, temperature=TEMP, top_p=TOP_P,
                        num_return_sequences=K,
                    )
                greedy_preds.extend(tokenizer.batch_decode(g, skip_special_tokens=True))
                decoded = tokenizer.batch_decode(s, skip_special_tokens=True)
                for j in range(len(batch)):
                    samples[i + j] = decoded[j * K : (j + 1) * K]
                if (i // 16) % 25 == 0:
                    print(f"[iter{it}] sampled {i + len(batch)}/{len(srcs)}", flush=True)

        winners: list[tuple[str, str]] = []
        for src, gold, gp, cands in zip(srcs, golds, greedy_preds, samples):
            gder = der(gp, gold)
            if gder == 0.0:
                continue
            scored = sorted(((der(c, gold), c) for c in cands))
            bder, best = scored[0]
            if bder < gder and bder <= KEEP_MAX_DER:
                winners.append((src, best))
        kept = len(winners)
        print(f"[iter{it}] kept {kept}/{len(srcs)} winner pairs", flush=True)
        if kept == 0:
            iter_marker.touch()
            checkpoints_volume.commit()
            continue

        args = Seq2SeqTrainingArguments(
            output_dir=str(raft_dir / f"iter{it}"),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            bf16=True,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            save_strategy="no",
            logging_steps=20,
            report_to=[],
            dataloader_num_workers=2,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=PairDataset(winners),
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        )
        trainer.train()
        model = trainer.model

        dev_der = mean_der(dev)
        print(f"[iter{it}] dev DER={dev_der:.4%} (SFT base={best_dev:.4%})", flush=True)

        if best_dev is None or dev_der < best_dev:
            best_dev = dev_der
            best_dir = raft_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            print(f"[iter{it}] new best -> {best_dir}", flush=True)

        import json

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "iter": it, "kept": kept, "prompts": len(srcs),
                "dev_der": dev_der, "best_dev": best_dev,
            }) + "\n")
        iter_marker.touch()
        checkpoints_volume.commit()

    # ---- final one-shot SadeedDiac-25 (Misraj evaluator) ----
    import pandas as pd
    import pyarrow.parquet as pq

    eval_model = model
    best_dir = raft_dir / "best"
    if best_dir.is_dir():
        eval_model = AutoModelForSeq2SeqLM.from_pretrained(str(best_dir)).to(device)
    eval_model.eval()

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    results: dict = {"run": RAFT_RUN, "best_dev": best_dev}
    for beam in (4, 1):
        preds: list[str] = []
        with torch.no_grad():
            for i in range(0, len(inputs), 16):
                batch = inputs[i : i + 16]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024,
                ).to(device)
                gen_kwargs = dict(max_new_tokens=1024)
                if beam > 1:
                    gen_kwargs.update(num_beams=beam)
                else:
                    gen_kwargs.update(num_beams=1)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = eval_model.generate(**enc, **gen_kwargs)
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
                if (i // 16) % 20 == 0:
                    print(f"[gen beam={beam}] {i + len(batch)}/{len(inputs)}", flush=True)

        from sadeed_evaluator import ArabicDiacritizationEvaluator as E

        csv_path = Path(f"/tmp/raft_sadeed_beam{beam}.csv")
        pd.DataFrame({"gt": outputs, "pred": preds}).to_csv(csv_path, index=False, header=False)
        print(f"\n===== RAFT beam={beam} (their default protocol) =====", flush=True)
        E.report_errors_on_csv_file(
            str(csv_path), ground_truth_column_index=0, predicted_column_index=1,
            has_header=False, gt_missing_diacritic_is_error=False,
        )
        (raft_dir / f"sadeed_preds_beam{beam}.csv").write_text(csv_path.read_text(), encoding="utf-8")
        checkpoints_volume.commit()

    done_marker.touch()
    checkpoints_volume.commit()
    return results


@app.local_entrypoint()
def main():
    run.remote()

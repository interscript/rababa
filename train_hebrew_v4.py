"""Hebrew v4: Train on ALL available labeled Hebrew data.

Sources combined:
- Nakdimon train (30K lines, mixed modern/Biblical)
- Sefaria Tanakh (15K lines, Biblical — matches test domain)
- DictaBERT-distilled (15K lines, distilled from modern Hebrew)
- Hebrew-expanded-v2 (22K lines, current combined)

Total: ~60K unique labeled examples (3x what v2 used).

Architecture: ByT5-base (same as v2)
Target: < 12% DER (vs v2's 17.3%)

Usage:
    modal run --detach train_hebrew_v3.py
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

app = modal.App(name=f"{APP_NAME}-hebrew-v4", image=image)


def _load_labeled_lines(path: Path) -> list[str]:
    """Load diacritized Hebrew lines from a file."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and any("֑" <= c <= "ׇ" for c in line):
            out.append(line)
    return out


_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def _strip_nikud(s: str) -> str:
    """Strip nikud only (vowels/dagesh), KEEP teamim — matches v2 format."""
    return "".join(c for c in s if c not in _NIKUD_MARKS)


def _has_consonants(s: str) -> bool:
    hebrew_consonants = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
    return any(c in hebrew_consonants for c in s)


@app.function(
    cpu=2,
    timeout=10 * 60,
    volumes={"/datasets": datasets_volume},
)
def build_combined_corpus() -> dict:
    """Combine all labeled Hebrew data into one corpus."""
    from pathlib import Path as _P

    datasets_volume.reload()

    sources = {
        "nakdimon": _P("/opt/rababa/data/nakdimon/train.txt"),
        "sefaria_tanakh": _P("/opt/rababa/data/sefaria-tanakh/train.txt"),
        "distilled_v1": _P("/opt/rababa/data/hebrew-distilled/train.txt"),
        "distilled_v2": _P("/opt/rababa/data/hebrew-dictabert-distilled/train.txt"),
        "expanded_v2": _P("/opt/rababa/data/hebrew-expanded-v2/train.txt"),
    }
    # Try also Modal-stored data
    modal_sources = {
        "nakdimon_m": _P("/datasets/nakdimon/train.txt"),
        "sefaria_m": _P("/datasets/sefaria/train.txt"),
        "distilled_m": _P("/datasets/hebrew-distilled/train.txt"),
        "dictabert_distilled_m": _P("/datasets/hebrew-dictabert-distilled/train.txt"),
        "expanded_v2_m": _P("/datasets/nakdimon-combined/train.txt"),
    }
    sources.update(modal_sources)

    all_pairs = []
    seen = set()
    counts = {}
    for name, path in sources.items():
        lines = _load_labeled_lines(path)
        new_count = 0
        for line in lines:
            undiacritized = _strip_nikud(line).strip()
            if not undiacritized or not _has_consonants(undiacritized):
                continue
            if len(undiacritized) < 5 or len(undiacritized) > 500:
                continue
            key = (undiacritized, line)
            if key in seen:
                continue
            seen.add(key)
            all_pairs.append({"src": undiacritized, "tgt": line})
            new_count += 1
        counts[name] = new_count
        print(f"[corpus] {name}: {new_count} unique", flush=True)

    print(f"[corpus] total unique pairs: {len(all_pairs)}", flush=True)

    # Write to datasets volume
    out_root = _P("/datasets/hebrew-v4")
    out_root.mkdir(parents=True, exist_ok=True)

    # Use nakdimon test as held-out test set
    test_path = _P("/opt/rababa/data/nakdimon/test.txt")
    if not test_path.is_file():
        test_path = _P("/datasets/nakdimon/test.txt")
    test_lines = _load_labeled_lines(test_path)
    test_pairs = []
    for line in test_lines:
        undiacritized = _strip_nikud(line).strip()
        if undiacritized and _has_consonants(undiacritized):
            test_pairs.append({"src": undiacritized, "tgt": line})

    # val split: take 5% of train
    import random
    rng = random.Random(42)
    rng.shuffle(all_pairs)
    n_val = max(500, len(all_pairs) // 20)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    for name, split in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        out = out_root / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split)} -> {out}", flush=True)

    datasets_volume.commit()
    return {"total_train": len(train_pairs), "total_val": len(val_pairs), "total_test": len(test_pairs), "sources": counts}


@app.function(
    gpu="A100",
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    """Train ByT5-base on Hebrew v4 corpus."""
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
        return {"error": "No training data. Run build_combined_corpus first."}

    print("[v3] loading ByT5-base...", flush=True)
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
    print(f"[v3] train={len(train_ds)}, val={len(val_ds)}", flush=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    ckpt_root = Path("/checkpoints/rababa_hebrew_byt5_v4/run-001")
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
        seed=42,
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
    return {"best": str(best_path), "n_train": len(train_ds)}


@app.function(
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def evaluate(num_beams: int = 1) -> dict:
    """Evaluate Hebrew v4 on Nakdimon test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    datasets_volume.reload()
    checkpoints_volume.reload()

    ckpt = Path("/checkpoints/rababa_hebrew_byt5_v4/run-001/best")
    if not ckpt.is_dir():
        return {"error": f"{ckpt} not found"}

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda")
    model.eval()

    test_path = Path("/datasets/hebrew-v4/test.jsonl")
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        s = (r.get("src") or "").strip()
        t = (r.get("tgt") or "").strip()
        if s and t:
            examples.append((s, t))

    print(f"[v3-eval] test examples: {len(examples)}", flush=True)

    total_wrong = 0
    total_chars = 0
    n_examples = 0
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            src = [s for s, _ in batch]
            gold = [g for _, g in batch]
            enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            gen = model.generate(**enc, max_new_tokens=512, num_beams=num_beams)
            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for pred, g in zip(preds, gold):
                wrong, total = _compare_diacritized(pred, g)
                total_wrong += wrong
                total_chars += total
                n_examples += 1

            if i == 0:
                for j in range(min(3, len(batch))):
                    print(f"--- Example {i+j} ---", flush=True)
                    print(f"  in:   {src[j]}", flush=True)
                    print(f"  pred: {preds[j]}", flush=True)
                    print(f"  gold: {gold[j]}", flush=True)

            if i % 640 == 0 and i > 0:
                der = total_wrong / max(1, total_chars)
                print(f"  [{i}/{len(examples)}] DER={der:.4f}", flush=True)

    der = total_wrong / max(1, total_chars)
    result = {"der": der, "n_examples": n_examples}
    print(f"=== Hebrew v4 DER: {der:.4f} ({n_examples} examples) ===", flush=True)
    return result


def _compare_diacritized(pred: str, gold: str) -> tuple[int, int]:
    """Count wrong consonant positions (those with mismatched diacritics)."""
    def _split(s):
        result = []
        cur_c = None
        cur_diacritics = []
        for c in s:
            if "֑" <= c <= "ׇ":
                cur_diacritics.append(c)
            else:
                if cur_c is not None:
                    result.append((cur_c, "".join(cur_diacritics)))
                cur_c = c
                cur_diacritics = []
        if cur_c is not None:
            result.append((cur_c, "".join(cur_diacritics)))
        return result

    p = _split(pred)
    g = _split(gold)
    if len(p) != len(g):
        return max(len(p), len(g)), max(len(p), len(g))
    wrong = sum(1 for a, b in zip(p, g) if a != b)
    return wrong, len(g)


@app.local_entrypoint()
def main():
    """Build corpus -> train -> evaluate."""
    corpus = build_combined_corpus.remote()
    print(f"Corpus: {json.dumps(corpus, indent=2)}")

    train_result = train.remote()
    print(f"Train: {json.dumps(train_result, indent=2, default=str)}")

    eval_result = evaluate.remote()
    print(f"Evaluate: {json.dumps(eval_result, indent=2)}")

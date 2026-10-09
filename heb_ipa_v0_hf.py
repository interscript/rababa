"""WO30 — Hebrew learned-IPA v0 (audio-supervised, FLEURS scale).

Corpus: FLEURS he_il (train+dev+test, ~5K utterances, CC-BY-4.0):
  transcript  (undiacritized Hebrew)
  phonemes    = universal phoneme CTC (wav2vec2-lv-60-espeak-cv-ft)
Student: byt5-small seq2seq text -> IPA (space-separated phones),
small-data regime: up to 60 epochs, early profile.
Gate: phonikud heb-g2p-benchmark CER < our rules layer (0.2393);
stretch: beat Nakdimon-class (0.1102). NO LLM labels anywhere.

Run on an apt-enabled job (espeak-ng):
  python /ckpt/run-032-heb-ipa-v0/heb_ipa_v0_hf.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, "/code")
sys.path.insert(0, "/code/src")

CKPT = Path("/ckpt/run-032-heb-ipa-v0")
GT = Path("/code/../data/heb-g2p/gt.tsv")  # not present; GT via env below
EPOCHS = 60
BS = 16

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download


def norm_target(ph: str) -> str:
    # espeak conventions, spaces between phones
    s = ph.replace("ˈ", "").replace("ˌ", "").replace(".", "").replace("-", "")
    return " ".join(s.split())


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    pairs_path = CKPT / "fleurst-ipa.jsonl"

    # ---- corpus (resumable) ----
    if not pairs_path.exists():
        from transformers import AutoModelForCTC, AutoProcessor

        proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        asr = AutoModelForCTC.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft").eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        asr = asr.to(dev)

        rows = []
        for split in ("train", "dev", "test"):
            tsv = hf_hub_download("google/fleurs", f"data/he_il/{split}.tsv",
                                  repo_type="dataset")
            with open(tsv, encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) >= 3:
                        rows.append((split, row))
        wanted = {}
        for split, row in rows:
            name = row[1] if row[1].endswith(".wav") else row[1] + ".wav"
            wanted.setdefault(split, {})[name] = row[2]

        pairs = []
        t0 = time.time()
        with pairs_path.open("w", encoding="utf-8") as out:
            for split in ("train", "dev", "test"):
                tar = hf_hub_download("google/fleurs", f"data/he_il/audio/{split}.tar.gz",
                                      repo_type="dataset")
                with tarfile.open(tar, "r:gz") as tf:
                    for m in tf.getmembers():
                        base = m.name.split("/")[-1]
                        if base not in wanted[split]:
                            continue
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        text = wanted[split][base].strip()
                        if not text:
                            continue
                        speech, sr = sf.read(io.BytesIO(f.read()))
                        if speech.ndim > 1:
                            speech = speech.mean(axis=1)
                        if len(speech) < 16000:
                            continue
                        inp = proc(speech.astype(np.float32), sampling_rate=sr,
                                   return_tensors="pt")
                        inp = {k: v.to(dev) for k, v in inp.items()}
                        with torch.no_grad():
                            logits = asr(**inp).logits
                        ph = norm_target(proc.batch_decode(logits.argmax(-1))[0])
                        if len(ph) < 8:
                            continue
                        pairs.append({"text": text, "ipa": ph, "split": split})
                        out.write(json.dumps(pairs[-1], ensure_ascii=False) + "\n")
        print(f"[corpus] {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
    else:
        pairs = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[corpus] resumed {len(pairs)} pairs", flush=True)

    train = [p for p in pairs if p["split"] == "train"]
    heldout = [p for p in pairs if p["split"] != "train"][:300]
    print(f"[data] train={len(train)} heldout={len(heldout)}", flush=True)

    # ---- student: byt5-small seq2seq text->IPA ----
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small").cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    class DS(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            return (train[i]["text"], train[i]["ipa"])

    def collate(b):
        # ByT5 tokens are BYTES: Hebrew chars are multi-byte — pad to
        # tokenized lengths, never char lengths
        es = [tok(t).input_ids[:-1] for t, _ in b]
        ds_ = [tok(p).input_ids for _, p in b]
        mx = max(len(e) for e in es)
        md = max(len(d) for d in ds_)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        am = torch.zeros((len(b), mx), dtype=torch.long)
        lb = torch.full((len(b), md), -100, dtype=torch.long)
        for i, (e, d) in enumerate(zip(es, ds_)):
            ids[i, :len(e)] = torch.tensor(e)
            am[i, :len(e)] = 1
            lb[i, :len(d)] = torch.tensor(d)
        return ids, am, lb

    loader = DataLoader(DS(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    total_steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 100, total_steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for ids, am, lb in loader:
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"[train] {step}/{total_steps} loss={float(out.loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    model.config.use_cache = True
    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    tok.save_pretrained(best)

    # ---- gate: phonikud heb-g2p-benchmark (their CER, their conventions) ----
    gt_path = Path("/ckpt/run-032-heb-ipa-v0/gt.tsv")
    gt = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        s, _, p = line.partition("\t")
        if s and p:
            gt.append((s, p))
    print(f"[gate] gt sentences={len(gt)}", flush=True)

    def lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
            prev = cur
        return prev[-1]

    preds = []
    with torch.no_grad():
        for i in range(0, len(gt), 16):
            batch = [s for s, _ in gt[i:i+16]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))

    cers, cers_ns = [], []
    for (s, gold), pred in zip(gt, preds):
        g = gold.replace("ˈ", "")
        p = pred.replace("ˈ", "")
        cers.append(lev(pred, gold) / max(1, len(gold)))
        cers_ns.append(lev(p, g) / max(1, len(g)))
    cer = sum(cers) / len(cers)
    cer_ns = sum(cers_ns) / len(cers_ns)
    verdict = {"model": "heb-ipa-v0", "cer": round(cer, 4), "cer_no_stress": round(cer_ns, 4),
               "gate": "cer < 0.2393 (rules layer)", "rules_baseline": 0.2393,
               "board": {"renikud": 0.0244, "phonikud": 0.0495, "nakdimon": 0.1102}}
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)
    for (s, gold), pred in list(zip(gt, preds))[:3]:
        print(f"  src: {s[:40]}\n  pred: {pred[:60]}\n  gold: {gold[:60]}", flush=True)


if __name__ == "__main__":
    main()

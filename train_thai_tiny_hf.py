# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "datasets>=3.0",
#   "accelerate>=1.1.0",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
#   "numpy<2",
# ]
# ///
"""WO08: Thai small-tier student — sequence-level KD into a ~12M model.

Pipeline in one job: stream a Thai Wikipedia slice, teacher-label with
B-K/umt5-thai-g2p-v2-0.5k (the published teacher lineage; NO LLM labels),
train a compact char encoder-decoder (d=256, 4+4 layers) on teacher
outputs, then gate on the kaikki test with the corpus-PER harness.

Gate (pre-registered): PER <= 3.5% (greedy, corpus-level, true
Levenshtein) AND int8 export < 10 MB. Ship only if both hold; the
artifact contract (kind=tiny-g2p) is designed after the gate.

Usage:
    hf jobs uv run --flavor a10g-large -d --timeout 12h \
      -v hf://datasets/Interscript/thai-g2p:/train_data:ro \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      train_thai_tiny_hf.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CKPT = Path("/ckpt/run-025-thai-tiny")
DATA = Path("/train_data")
TEACHER = "B-K/umt5-thai-g2p-v2-0.5k"
N_PARAS = 200_000
MAX_SRC = 300
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
D_FF = 1024
EPOCHS = 4
BS = 128
GATE_PER = 3.5
GATE_MB = 10.0
PER = 100.0


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def corpus_per(pairs: list[tuple[str, str]]) -> float:
    dist = sum(lev(p, t) for p, t in pairs)
    total = sum(len(t) for _, t in pairs)
    return PER * dist / max(1, total)


def build_corpus() -> list[tuple[str, str]]:
    """teacher-labeled pairs: kaikki test-side srcs + a wiki slice."""
    pairs: list[tuple[str, str]] = []
    import pandas as pd

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16).to(device).eval()

    srcs: list[str] = []
    test_path = DATA / "kaikki-test.jsonl"
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            srcs.append(json.loads(line)["src"])
    print(f"[data] kaikki srcs={len(srcs)}", flush=True)

    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train", streaming=True)
    n = 0
    for row in ds:
        for para in row["text"].split("\n"):
            p = para.strip()
            if 10 <= len(p) <= MAX_SRC:
                srcs.append(p)
                n += 1
        if n >= N_PARAS:
            break
    print(f"[data] total srcs={len(srcs)}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(srcs), 64):
            batch = srcs[i:i + 64]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SRC).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = teacher.generate(**enc, max_new_tokens=MAX_SRC, num_beams=1)
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            for s, t in zip(batch, outs):
                if t.strip():
                    pairs.append((s, t))
            if (i // 64) % 100 == 0:
                print(f"[teacher] {i + len(batch)}/{len(srcs)} "
                      f"({(i + len(batch)) / (time.time() - t0):.0f} src/s)", flush=True)
    return pairs


class TinyG2P(nn.Module):
    """Compact char encoder-decoder with a small IPA vocab."""

    def __init__(self, vocab: dict[str, int]):
        super().__init__()
        self.vocab = vocab
        n_v = len(vocab)
        self.emb = nn.Embedding(n_v, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, D_FF, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        dlayer = nn.TransformerDecoderLayer(
            D_MODEL, N_HEADS, D_FF, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dlayer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, n_v)

    def forward(self, src, tgt_in, src_pad, tgt_pad):
        mem = self.enc(self.emb(src), src_key_padding_mask=src_pad)
        mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=src.device)
        h = self.dec(self.emb(tgt_in), mem, tgt_mask=mask,
                     memory_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
        return self.head(h)


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    pairs_path = CKPT / "distill.jsonl"
    if pairs_path.exists():
        pairs = [(json.loads(l)["src"], json.loads(l)["tgt"])
                 for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        pairs = build_corpus()
        with pairs_path.open("w", encoding="utf-8") as f:
            for s, t in pairs:
                f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")
    print(f"[data] pairs={len(pairs)}", flush=True)

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for s, t in pairs:
        for ch in s + t:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    test = [(json.loads(l)["src"], json.loads(l)["tgt"])
            for l in (DATA / "kaikki-test.jsonl").read_text().splitlines() if l.strip()]
    for s, t in test:
        for ch in s + t:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    print(f"[vocab] {len(vocab)}", flush=True)

    # kaikki-test srcs must NOT be in training pairs (contamination guard)
    test_srcs = {s for s, _ in test}
    pairs = [(s, t) for s, t in pairs if s not in test_srcs]

    model = TinyG2P(vocab).cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.1f}M", flush=True)

    pad, bos, eos = vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]

    class DS(Dataset):
        def __getitem__(self, i):
            s, t = pairs[i]
            return ([vocab[c] for c in s], [bos] + [vocab[c] for c in t],
                    [vocab[c] for c in t] + [eos])

        def __len__(self):
            return len(pairs)

    def collate(b):
        ms = max(len(x[0]) for x in b)
        mt = max(len(x[1]) for x in b)
        src = torch.full((len(b), ms), pad, dtype=torch.long)
        tin = torch.full((len(b), mt), pad, dtype=torch.long)
        tout = torch.full((len(b), mt), pad, dtype=torch.long)
        for i, (s, ti, to) in enumerate(b):
            src[i, :len(s)] = torch.tensor(s)
            tin[i, :len(ti)] = torch.tensor(ti)
            tout[i, :len(to)] = torch.tensor(to)
        return src, tin, tout

    loader = DataLoader(DS(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total = len(loader) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    ce = nn.CrossEntropyLoss(ignore_index=pad)

    step = 0
    for _ep in range(EPOCHS):
        for src, tin, tout in loader:
            src, tin, tout = src.cuda(), tin.cuda(), tout.cuda()
            sp = src == pad
            tp = tin == pad
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(src, tin, sp, tp)
            loss = ce(logits.reshape(-1, len(vocab)), tout.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"[train] {step}/{total} loss={float(loss):.4f}", flush=True)

    # ---- greedy decode kaikki-test, corpus PER ----
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            batch = [s for s, _ in test[i:i + 64]]
            ms = max(len(s) for s in batch)
            src = torch.full((len(batch), ms), pad, dtype=torch.long)
            for r, s in enumerate(batch):
                ids = [vocab[c] for c in s]
                src[r, :len(ids)] = torch.tensor(ids)
            src = src.cuda()
            mem = model.enc(model.emb(src), src_key_padding_mask=(src == pad))
            ys = torch.full((len(batch), 1), bos, dtype=torch.long, device="cuda")
            done = torch.zeros(len(batch), dtype=torch.bool)
            for _ in range(MAX_SRC):
                mask = nn.Transformer.generate_square_subsequent_mask(
                    ys.size(1), device="cuda")
                h = model.dec(model.emb(ys), mem, tgt_mask=mask,
                              memory_key_padding_mask=(src == pad))
                nxt = h[:, -1].argmax(-1)
                nxt[done] = pad
                ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
                done |= nxt == eos
                if done.all():
                    break
            for row in ys[:, 1:].tolist():
                toks = [c for c in row if c not in (pad, eos)]
                preds.append("".join(inv[t] for t in toks))
    inv = {v: k for k, v in vocab.items()}
    per = corpus_per(list(zip(preds, [t for _, t in test])))
    print(f"[gate] val PER (greedy, corpus) = {per:.4f}% gate<={GATE_PER}%", flush=True)

    # int8 export size check
    fp = CKPT / "tiny-fp32.onnx"
    src0 = torch.zeros((1, 8), dtype=torch.long)
    tin0 = torch.zeros((1, 4), dtype=torch.long)
    sp0 = torch.zeros((1, 8), dtype=torch.bool)
    tp0 = torch.zeros((1, 4), dtype=torch.bool)
    torch.onnx.export(model, (src0, tin0, sp0, tp0), str(fp),
                      input_names=["src", "tgt_in", "src_pad", "tgt_pad"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = CKPT / "tiny-int8.onnx"
    quantize_dynamic(str(fp), str(q), weight_type=QuantType.QInt8)
    mb = q.stat().st_size / 1e6
    verdict = {"per": round(per, 4), "int8_mb": round(mb, 1), "params_m": round(n_params / 1e6, 1),
               "gate_per": GATE_PER, "gate_mb": GATE_MB,
               "passed": bool(per <= GATE_PER and mb < GATE_MB)}
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)
    (CKPT / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

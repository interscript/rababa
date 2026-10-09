"""WO32 — their literal architecture: BiLSTM haraqat tagger at full
silver scale (news-pure, 900K silver cap), the QCRI recipe class.

Fadel/QCRI-style: char embeddings -> stacked BiLSTM -> per-position
haraqat classifier over the plane-combo inventory. No pretrained
backbone — from-scratch, exactly their class. The SOTA-stack layer
lives in run-033 (byt5-large plane) above it.

Dual gate: SadeedDiac-25 windowed zero-skip + WikiNews-2024 multiref
(their 2.70 is the ID+OOD target class).

Usage:
    python /ckpt/run-034-bilstm/train_bilstm_hf.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/code")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import haraqat_planes as HP

DATA = Path("/train_data")
RUN = os.environ.get("BIL_RUN", "run-034-bilstm")
CKPT = Path("/ckpt") / RUN
SILVER_CAP = int(os.environ.get("BIL_SILVER_CAP", "900000"))
EPOCHS = int(os.environ.get("BIL_EPOCHS", "6"))
BS = int(os.environ.get("BIL_BS", "64"))
EMB = 128
HID = 512
LAYERS = 3
WINDOW = 1400

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def split_windows(text: str, budget: int = WINDOW) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words, cur, n, wins = text.split(), [], 0, []
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # ---- corpus: news-pure + full silver (identical to run-033) ----
    lines: list[str] = []
    news = [l for l in (DATA / "news.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [l for l in (DATA / "wikinews2014_gold.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    units = news + gold * 4
    silver = 0
    articles = (DATA / "qcri-wiki.jsonl").read_text(encoding="utf-8").splitlines()
    rng.shuffle(articles)
    for aline in articles:
        if silver >= SILVER_CAP:
            break
        if not aline.strip():
            continue
        art = json.loads(aline).get("text", "")
        for para in art.split("\n"):
            para = para.strip()
            if not para:
                continue
            for w in split_windows(para):
                units.append(w)
                silver += 1
            if silver >= SILVER_CAP:
                break
    print(f"[data] news={len(news)} gold={len(gold)} silver={silver}", flush=True)

    # defense: drop units whose SKELETON is empty (pure-mark windows
    # would crash/produce nothing) — mirrors make_unit in the plane trainer
    units = [u for u in (u.strip() for u in units)
             if u and DIACRITICS_RE.sub("", u).encode("utf-8") != b""]
    random.Random(42).shuffle(units)
    classes, counts = HP.plane_classes(units)
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    print(f"[data] units={len(units)} classes={n_classes}", flush=True)

    # char vocab from corpus skeletons
    vocab = {"<pad>": 0, "<unk>": 1}
    for u in units:
        for ch in HP.split_planes(u)[0]:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    print(f"[vocab] {len(vocab)}", flush=True)

    def encode(text: str):
        skel, labels = HP.split_planes(text)
        ids = [vocab.get(ch, 1) for ch in skel]
        lab = [0] * len(ids)
        for i, (ch, cluster) in enumerate(zip(skel, labels)):
            lab[i] = class_to_id.get(HP.canon_combo(cluster), 0)
        return ids, lab

    class DS(Dataset):
        def __len__(self):
            return len(units)

        def __getitem__(self, i):
            return encode(units[i])

    def collate(b):
        mx = max(len(x[0]) for x in b)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        lab = torch.full((len(b), mx), -100, dtype=torch.long)  # mask pads
        for i, (x, y) in enumerate(b):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
        return ids, lab

    loader = DataLoader(DS(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    total_steps = len(loader) * EPOCHS

    class BiLSTMTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), EMB, padding_idx=0)
            self.lstm = nn.LSTM(EMB, HID, num_layers=LAYERS, bidirectional=True,
                                batch_first=True, dropout=0.3)
            self.head = nn.Linear(HID * 2, n_classes)

        def forward(self, ids):
            h, _ = self.lstm(self.emb(ids))
            return self.head(h)

    model = BiLSTMTagger().cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.1f}M steps={total_steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for ids, lab in loader:
            ids, lab = ids.cuda(), lab.cuda()
            logits = model(ids)  # fp32: LSTMs NaN-collapse under bf16 autocast
            loss = ce(logits.reshape(-1, n_classes), lab.reshape(-1))
            if not torch.isfinite(loss):
                print(f"[nan] step {step}: non-finite loss, skipping batch", flush=True)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"[train] {step}/{total_steps} loss={float(loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes,
                "vocab": vocab}, best / "model.pt")

    # ---- gates ----
    import pandas as pd
    import pyarrow.parquet as pq

    @torch.no_grad()
    def tag(text: str) -> str:
        skel, _ = HP.split_planes(text)
        ids = [vocab.get(ch, 1) for ch in skel] or [0]
        t = torch.tensor([ids], device="cuda")
        logits = model(t)
        preds = logits[0].argmax(-1).tolist()
        return "".join(ch + (classes[preds[i]] if i < len(preds) else "")
                       for i, ch in enumerate(skel))

    table = pq.read_table(DATA / "sadeeddiac-25.parquet")
    s_inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    s_outputs = table.column("output").to_pylist()
    all_windows, counts = [], []
    for text in s_inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    preds = [tag(w) for w in all_windows]
    k = 0
    paras = []
    for _t, c in zip(s_inputs, counts):
        paras.append(" ".join(preds[k:k+c])); k += c
    csv_p = Path("/tmp/bilstm_sadeed.csv")
    pd.DataFrame({"gt": s_outputs, "pred": paras}).to_csv(csv_p, index=False, header=False)
    (CKPT / "sadeed_preds.csv").write_text(csv_p.read_text(), encoding="utf-8")

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"===== {RUN} SadeedDiac-25 (QCRI-target class) =====", flush=True)
    E.report_errors_on_csv_file(str(csv_p), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)

    import eval_wikinews_multiref as WN

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    ww, wc = [], []
    for text in winputs:
        ws = split_windows(text); wc.append(len(ws)); ww.extend(ws)
    wpreds = [tag(w) for w in ww]
    k = 0; wparas = []
    for _t, c in zip(winputs, wc):
        wparas.append(" ".join(wpreds[k:k+c])); k += c
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": RUN, "surface": "wikinews-2024-multiref", **wscore},
                     ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref.json").write_text(
        json.dumps({"run": RUN, **wscore}, ensure_ascii=False, indent=1))
    (CKPT / "EVAL_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

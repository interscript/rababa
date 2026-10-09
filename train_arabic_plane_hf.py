# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "accelerate>=1.1.0",
#   "pandas",
#   "pyarrow",
#   "pyarabic",
#   "prettytable",
# ]
# ///
"""WO21 — Arabic plane register transfer (HF port of run-018 v2).

run-018 recipe (byt5-small encoder + plane embedding + per-position
head, K=2 mask-predict, benchmark-convention running text) with the
WO17 silver mixture knob: QCRI Wikipedia silver windowed into units at
PLANE_SILVER_CAP (0 = pure run-018 reproduction).

Dual-surface gate (both in-job): SadeedDiac-25 windowed zero-skip
(rung gate 4.5701 / transfer gate: hold ≤2.7397) AND WikiNews-2024
multiref (plane-large baseline 18.69/11.35 — move it).

Usage:
    hf jobs uv run --flavor a100-large -d --timeout 24h \\
      -e PLANE_RUN=run-028-plane-transfer -e PLANE_SILVER_CAP=60000 \\
      --secrets HF_TOKEN \\
      -v hf://datasets/Interscript/arabic-r8:/train_data:ro \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -v /tmp/isx-hf-code:/code:ro \\
      train_arabic_plane_hf.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "/code")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

import haraqat_planes as HP

DATA = Path("/train_data")
RUN = os.environ.get("PLANE_RUN", "run-028-plane-transfer")
SILVER_CAP = int(os.environ.get("PLANE_SILVER_CAP", "0"))
BACKBONE = os.environ.get("PLANE_BACKBONE", "google/byt5-small")
MAX_LINES = int(os.environ.get("PLANE_MAX_LINES", "600000"))
EPOCHS = int(os.environ.get("PLANE_EPOCHS", "2"))
K_PASSES = int(os.environ.get("PLANE_K", "2"))
MICRO_BS = int(os.environ.get("PLANE_BS", "16"))
REGIME = os.environ.get("PLANE_REGIME", "full")  # full | news-pure
NEWS_UPSAMPLE = 3
GOLD_UPSAMPLE = 4
UNIT_BYTES = 1400
N_VAL = 2_000

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
CKPT = Path("/ckpt") / RUN


def make_unit(text: str) -> str | None:
    t = text.strip()
    if not t or DIACRITICS_RE.sub("", t).encode("utf-8") == b"":
        return None
    if len(t.encode("utf-8")) > 1450:
        return None
    return t


def split_windows(text: str, budget: int = UNIT_BYTES) -> list[str]:
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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    CKPT.mkdir(parents=True, exist_ok=True)
    done_marker = CKPT / "EVAL_DONE"
    prog_name = "eval_progress_EVAL_DONE.jsonl"

    # ---- corpus: benchmark-convention running text + optional silver ----
    lines: list[str] = []
    base_files = ("domain.txt", "replay.txt", "tashkeela-scale.txt") \
        if REGIME == "full" else ()
    for name in base_files:
        p = DATA / name
        if p.exists():
            lines += [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    news = [l for l in (DATA / "news.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [l for l in (DATA / "wikinews2014_gold.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    units = lines + news * NEWS_UPSAMPLE + gold * GOLD_UPSAMPLE

    if SILVER_CAP > 0:
        rng = random.Random(7)
        articles = (DATA / "qcri-wiki.jsonl").read_text(encoding="utf-8").splitlines()
        rng.shuffle(articles)
        silver = 0
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
                for w in split_windows(para, UNIT_BYTES - 50):
                    units.append(w)
                    silver += 1
                if silver >= SILVER_CAP:
                    break
        print(f"[data] silver units={silver}", flush=True)

    random.Random(42).shuffle(units)
    units = [u for u in (make_unit(u) for u in units[:MAX_LINES]) if u]
    print(f"[data] units={len(units)}", flush=True)

    # ---- inventory ----
    inv_path = CKPT / "inventory.json"
    if inv_path.exists():
        classes = json.loads(inv_path.read_text(encoding="utf-8"))
    else:
        classes, counts = HP.plane_classes(units)
        inv_path.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
        print(f"[inventory] {len(classes)} classes (incl. none)", flush=True)
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    MASK_ID = n_classes
    import hashlib

    labels_digest = hashlib.sha256(inv_path.read_bytes()).hexdigest()[:12]

    tok = AutoTokenizer.from_pretrained(BACKBONE)

    def encode_unit(text: str):
        skel, labels = HP.split_planes(text)
        ids = tok(skel).input_ids[:-1]
        label_ids = [0] * len(ids)
        pos = 0
        for ch, lab in zip(skel, labels):
            n = len(ch.encode("utf-8"))
            lab = lab.split("\x00", 1)[-1]
            cid = class_to_id.get(HP.canon_combo(lab), 0)
            for t in range(pos, min(pos + n, len(label_ids))):
                label_ids[t] = cid
            pos += n
        return ids, label_ids

    class PlaneDS(Dataset):
        def __len__(self):
            return len(units)

        def __getitem__(self, i):
            return encode_unit(units[i])

    def collate(batch):
        mx = max(len(x[0]) for x in batch)
        ids = torch.zeros((len(batch), mx), dtype=torch.long)
        lab = torch.zeros((len(batch), mx), dtype=torch.long)
        am = torch.zeros((len(batch), mx), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            lab[i, : len(y)] = torch.tensor(y)
            am[i, : len(x)] = 1
        return ids, am, lab

    val_rows = units[-N_VAL:]
    train_rows = units[:-N_VAL]
    train_loader = DataLoader(PlaneDS(), batch_size=MICRO_BS, shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=True)
    total_steps = len(train_loader) * EPOCHS
    print(f"[data] train={len(train_rows)} val={len(val_rows)} steps={total_steps} "
          f"classes={n_classes}", flush=True)

    backbone = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE)
    encoder = backbone.encoder.cuda().train()
    if os.environ.get("PLANE_CHECKPOINT") == "1":
        encoder.gradient_checkpointing_enable()
        encoder.config.use_cache = False
    d_model = encoder.config.d_model
    plane_emb = nn.Embedding(n_classes + 1, d_model).cuda().train()
    nn.init.normal_(plane_emb.weight, std=0.02)
    head = nn.Linear(d_model, n_classes).cuda().train()

    params = list(encoder.parameters()) + list(plane_emb.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, total_steps)
    ce = nn.CrossEntropyLoss()

    def usable(ck: Path) -> bool:
        m = ck / "labels.sha"
        return m.exists() and m.read_text().strip() == labels_digest

    ckpts = sorted((c for c in CKPT.glob("step-*") if usable(c)),
                   key=lambda p: int(p.name.split("-")[1]))
    step = 0
    if ckpts:
        sd = torch.load(ckpts[-1] / "model.pt", map_location="cpu", weights_only=True)
        encoder.load_state_dict(sd["encoder"])
        plane_emb.load_state_dict(sd["plane_emb"])
        head.load_state_dict(sd["head"])
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt", map_location="cpu", weights_only=True))
        step = int(ckpts[-1].name.split("-")[1])
        for _ in range(step):
            scheduler.step()
        print(f"[resume] step-{step}", flush=True)

    def forward_logits(ids, am, plane_in):
        embeds = encoder.embed_tokens(ids) + plane_emb(plane_in)
        out = encoder(inputs_embeds=embeds, attention_mask=am)
        return head(out.last_hidden_state)

    def save(s: int) -> None:
        ck = CKPT / f"step-{s}"
        ck.mkdir(exist_ok=True)
        (ck / "labels.sha").write_text(labels_digest)
        torch.save({"encoder": encoder.state_dict(),
                    "plane_emb": plane_emb.state_dict(),
                    "head": head.state_dict()}, ck / "model.pt")
        torch.save(optimizer.state_dict(), ck / "optim.pt")

    encoder.eval(); plane_emb.eval(); head.eval()

    @torch.no_grad()
    def predict_window(skel: str) -> str:
        ids, _ = encode_unit(skel)
        t_ids = torch.tensor([ids], device="cuda")
        am = torch.ones_like(t_ids)
        plane = torch.full_like(t_ids, MASK_ID)
        preds = None
        for _ in range(K_PASSES):
            with torch.autocast("cuda", torch.bfloat16):
                logits = forward_logits(t_ids, am, plane)
            preds = logits.argmax(-1)
            plane = preds
        pos = 0
        out = []
        for ch in skel:
            n = len(ch.encode("utf-8"))
            votes = preds[0, pos:pos + n].tolist()
            cid = max(set(votes), key=votes.count)
            out.append(ch + (classes[cid] if cid < n_classes else ""))
            pos += n
        return "".join(out)

    encoder.train(); plane_emb.train(); head.train()

    optimizer.zero_grad()
    micro = 0
    for _epoch in range(EPOCHS):
        for ids, am, lab in train_loader:
            if step >= total_steps:
                break
            ids, am, lab = ids.cuda(), am.cuda(), lab.cuda()
            q = random.random()
            keep = (torch.rand_like(lab, dtype=torch.float) >= q).long() * am
            plane_in = torch.where(keep == 1, lab, torch.full_like(lab, MASK_ID))
            with torch.autocast("cuda", torch.bfloat16):
                logits = forward_logits(ids, am, plane_in)
            loss = ce(logits.float().reshape(-1, n_classes), lab.reshape(-1))
            loss.backward()
            micro += 1
            if micro % 1 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 100 == 0:
                    print(f"[{RUN} {step}/{total_steps}] ce={float(loss):.4f}", flush=True)
                if step % 1000 == 0:
                    save(step)

    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "plane_emb": plane_emb.state_dict(),
                "head": head.state_dict(),
                "classes": classes}, best / "model.pt")
    encoder.eval(); plane_emb.eval(); head.eval()
    print("[ckpt] best saved", flush=True)

    # ---- surface 1: SadeedDiac-25 windowed zero-skip ----
    table = pq.read_table(DATA / "sadeeddiac-25.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()
    all_windows, counts = [], []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(inputs)} paragraphs -> {len(all_windows)} windows", flush=True)

    prog = CKPT / prog_name
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["i"]] = row["pred"]
    missing = [i for i in range(len(all_windows)) if i not in saved]
    with prog.open("a", encoding="utf-8") as out:
        for n_done, i in enumerate(missing, 1):
            saved[i] = predict_window(all_windows[i])
            out.write(json.dumps({"i": i, "pred": saved[i]}, ensure_ascii=False) + "\n")
            if n_done % 200 == 0:
                out.flush()
                print(f"[gen] {len(saved)}/{len(all_windows)}", flush=True)

    k = 0
    paragraphs = []
    for _t, c in zip(inputs, counts):
        paragraphs.append(" ".join(saved[i] for i in range(k, k + c)))
        k += c
    csv_path = Path("/tmp/sadeed_plane.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (CKPT / "sadeed_preds.csv").write_text(csv_path.read_text(), encoding="utf-8")

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"===== {RUN} SadeedDiac-25 windowed zero-skip "
          f"(transfer gate: hold <=2.7397; rung 4.5701) =====", flush=True)
    E.report_errors_on_csv_file(str(csv_path), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)

    # ---- surface 2: WikiNews-2024 multiref ----
    import eval_wikinews_multiref as WN

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    wwindows, wcounts = [], []
    for text in winputs:
        ws = split_windows(text)
        wcounts.append(len(ws))
        wwindows.extend(ws)
    wprog = CKPT / "eval_progress_WN.jsonl"
    wsaved: dict[int, str] = {}
    if wprog.exists():
        for line in wprog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                wsaved[row["i"]] = row["pred"]
    wmissing = [i for i in range(len(wwindows)) if i not in wsaved]
    with wprog.open("a", encoding="utf-8") as out:
        for n_done, i in enumerate(wmissing, 1):
            wsaved[i] = predict_window(wwindows[i])
            out.write(json.dumps({"i": i, "pred": wsaved[i]}, ensure_ascii=False) + "\n")
            if n_done % 100 == 0:
                out.flush()
                print(f"[wgen] {len(wsaved)}/{len(wwindows)}", flush=True)
    wparas = []
    k = 0
    for _t, c in zip(winputs, wcounts):
        wparas.append(" ".join(wsaved[i] for i in range(k, k + c)))
        k += c
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": RUN, "surface": "wikinews-2024-multiref",
                      "baseline_plane_large": "18.69/11.35", **wscore},
                     ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref.json").write_text(
        json.dumps({"run": RUN, **wscore}, ensure_ascii=False, indent=1))
    done_marker.touch()
    print("[done] EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

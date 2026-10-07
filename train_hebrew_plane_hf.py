# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "accelerate>=1.1.0",
# ]
# ///
"""run-022 — Hebrew plane scale-up on HF Jobs (WO04).

run-021 recipe (byt5-small, 2ep, K=2, 12.48 DER) scaled to byt5-base,
4 epochs, K=3. Data: hf://datasets/Interscript/hebrew-v4 (ro mount).
Checkpoints: hf://buckets/Interscript/isx-training (rw mount).

Gate (pre-registered, nakdimon test via /data/test.jsonl, greedy,
seq2seq_der, the exact run-021 protocol): DER <= 10.0 (shipped 1.0 is
12.48). Resume contract: labels.sha + step dirs + per-marker eval
progress + original-separator stitching (run-021 lessons intact).

Usage:
    hf jobs uv run --flavor a100-large -d --timeout 24h \
      -v hf://datasets/Interscript/hebrew-v4:/train_data:ro \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v /tmp/isx-hf-code:/code:ro \
      train_hebrew_plane_hf.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "/code")
sys.path.insert(0, "/code/src")

DATA = Path("/train_data")
CKPT = Path("/ckpt")
RUN = "run-022-heb-plane-base"
UNIT_BYTES = 1400
N_VAL = 2_000
EPOCHS = 4
MAX_LINES = 600_000
K_PASSES = 3
MICRO_BS = 8
GRAD_ACCUM = 2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

import nikud_planes as NP
from rababa.evaluate import seq2seq_der


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    run_dir = CKPT / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "EVAL_DONE"
    prog_name = f"eval_progress_{done_marker.name}.jsonl"

    units = []
    for row in load_jsonl(DATA / "train.jsonl"):
        t = row["tgt"].strip()
        if not t or len(t.encode("utf-8")) > 1450:
            continue
        units.append(t)
    random.Random(42).shuffle(units)
    units = units[:MAX_LINES]
    print(f"[data] units={len(units)}", flush=True)

    inv_path = run_dir / "inventory.json"
    if inv_path.exists():
        classes = json.loads(inv_path.read_text(encoding="utf-8"))
    else:
        from collections import Counter

        counts: Counter = Counter()
        for u in units:
            _, cls = NP.split_planes(u)
            counts.update(c for c in cls if c)
        classes = [""] + sorted(counts)
        inv_path.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
        print(f"[inventory] {len(classes)} classes (incl. none)", flush=True)
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    MASK_ID = n_classes
    import hashlib

    labels_digest = hashlib.sha256(inv_path.read_bytes()).hexdigest()[:12]

    tok = AutoTokenizer.from_pretrained("google/byt5-base")

    def encode_unit(text: str):
        skel, labels = NP.split_planes(text)
        ids = tok(skel).input_ids[:-1]
        label_ids = [0] * len(ids)
        pos = 0
        for ch, lab in zip(skel, labels):
            n = len(ch.encode("utf-8"))
            cid = class_to_id.get(lab, 0)
            for t in range(pos, min(pos + n, len(label_ids))):
                label_ids[t] = cid
            pos += n
        return ids, label_ids

    class PlaneDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return encode_unit(self.rows[i])

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
    train_units = units[:-N_VAL]
    train_loader = DataLoader(
        PlaneDS(train_units), batch_size=MICRO_BS, shuffle=True, collate_fn=collate,
        num_workers=2, drop_last=True,
    )
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    print(f"[data] train={len(train_units)} val={len(val_rows)} steps={total_steps} "
          f"classes={n_classes}", flush=True)

    backbone = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")
    encoder = backbone.encoder.cuda().train()
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

    ckpts = sorted((c for c in run_dir.glob("step-*") if usable(c)),
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

    def save_ckpt(s: int) -> None:
        ck = run_dir / f"step-{s}"
        ck.mkdir(exist_ok=True)
        (ck / "labels.sha").write_text(labels_digest)
        torch.save({"encoder": encoder.state_dict(),
                    "plane_emb": plane_emb.state_dict(),
                    "head": head.state_dict()}, ck / "model.pt")
        torch.save(optimizer.state_dict(), ck / "optim.pt")
        print(f"[ckpt] step-{s} saved", flush=True)

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
            loss = ce(logits.float().reshape(-1, n_classes), lab.reshape(-1)) / GRAD_ACCUM
            loss.backward()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"[{RUN} step {step}/{total_steps}] ce={float(loss) * GRAD_ACCUM:.4f} q={q:.2f}", flush=True)
                if step % 1000 == 0:
                    save_ckpt(step)

    best = run_dir / "best"
    best.mkdir(exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "plane_emb": plane_emb.state_dict(),
                "head": head.state_dict(),
                "classes": classes}, best / "model.pt")
    print("[ckpt] best saved", flush=True)

    # ---- gate: nakdimon test (/data/test.jsonl), K-pass, seq2seq_der ----
    encoder.eval()
    plane_emb.eval()
    head.eval()

    targets = [r["tgt"] for r in load_jsonl(DATA / "test.jsonl")]
    all_windows: list[str] = []
    counts_w: list[int] = []
    for text in targets:
        ws = split_windows(text.strip())
        counts_w.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(targets)} examples -> {len(all_windows)} windows", flush=True)

    prog = run_dir / prog_name
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[gen] resuming with {len(saved)} saved windows", flush=True)

    @torch.no_grad()
    def predict_window(text: str) -> str:
        skel, _ = NP.split_planes(text)
        ids = tok(skel).input_ids[:-1]
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
        char_preds: list[str] = []
        for ch in skel:
            n = len(ch.encode("utf-8"))
            votes = preds[0, pos : pos + n].tolist()
            cid = max(set(votes), key=votes.count)
            char_preds.append(classes[cid] if cid < n_classes else "")
            pos += n
        return NP.render(skel, char_preds)

    missing = [i for i in range(len(all_windows)) if i not in saved]
    with prog.open("a", encoding="utf-8") as prog_out:
        for n_done, i in enumerate(missing, 1):
            pred = predict_window(all_windows[i])
            prog_out.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
            saved[i] = pred
            if n_done % 200 == 0:
                prog_out.flush()
                print(f"[gen] {len(saved)}/{len(all_windows)}", flush=True)
    prog.write_text(prog.read_text(encoding="utf-8"), encoding="utf-8")  # ensure flushed

    # stitch with ORIGINAL separators (run-021 lesson)
    k = 0
    total_wrong = 0.0
    total_positions = 0
    for tgt, c in zip(targets, counts_w):
        text = tgt.strip()
        words = text.split()
        seps = re.findall(r"\s+", text)
        sep_for = {i: (seps[i] if i < len(seps) else "") for i in range(len(words))}
        rebuilt: list[str] = []
        for _g in range(c):
            pred = saved[k]; k += 1
            for w in pred.split():
                wi = sum(len(part.split()) for part in rebuilt)
                rebuilt.append(w + sep_for.get(wi, " "))
        pred = "".join(rebuilt)
        d, n = seq2seq_der(pred, tgt)
        total_wrong += d * n
        total_positions += n

    der = total_wrong / max(1, total_positions)
    verdict = {"run": RUN, "steps": step, "k_passes": K_PASSES, "der": der,
               "gate": 10.0, "baseline_1_0": 12.48}
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print(f"\n===== {RUN} nakdimon greedy DER: {der:.4f} "
          f"(gate <= 10.0; heb-diac-plane-1.0 = 12.48) =====", flush=True)
    done_marker.touch()
    print("[done] EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

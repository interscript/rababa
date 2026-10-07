# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "accelerate>=1.1.0",
#   "pyarrow",
# ]
# ///
"""WO06 stage 1: r7 teacher pseudo-labels arwiki (noisy student).

NO LLM LABELS — the teacher is our own r7 dedicated model (run-007-news).
Greedy decode (the protocol decode) + a teacher-forced margin pass;
windows are kept when a high fraction of decoded tokens are confident
(top1-top2 logit margin > 1.0 nat). Output: arwiki-pseudo.jsonl.

Usage:
    hf jobs uv run --flavor a10g-large -d --timeout 8h \
      -v hf://datasets/Interscript/arabic-r8:/train_data:ro \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      label_arwiki_hf.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DATA = Path("/train_data")
CKPT = Path("/ckpt/r8-pseudo")
TEACHER = Path("/ckpt/r7-best")
UNIT_BYTES = 1400
MARGIN_NAT = 1.0
KEEP_FRAC = 0.9
MAX_UNITS = 12_000


def window(text: str, budget: int = UNIT_BYTES) -> list[str]:
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
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16).to(device).eval()

    units: list[str] = []
    for name in ("train.txt", "val.txt"):
        p = DATA / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            units.extend(window(line))
            if len(units) >= MAX_UNITS:
                break
        if len(units) >= MAX_UNITS:
            break
    print(f"[data] arwiki units={len(units)}", flush=True)

    out_path = CKPT / "arwiki-pseudo.jsonl"
    done = 0
    kept = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out, torch.no_grad():
        for i in range(0, len(units), 8):
            batch = units[i:i + 8]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=1450).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=1500, num_beams=1,
                                     return_dict_in_generate=True, output_scores=True)
            seqs = gen.sequences
            texts = tok.batch_decode(seqs, skip_special_tokens=True)

            # margins straight from generate's per-step scores (one pass)
            dec_len = seqs.size(1) - 1
            if dec_len > 0 and len(gen.scores) == dec_len:
                logits = torch.stack(gen.scores, dim=1).float()  # [B, T, V]
                top2 = logits.topk(2, dim=-1).values
                margins = top2[..., 0] - top2[..., 1]
                real = seqs[:, 1:] != tok.pad_token_id
                conf = (margins > MARGIN_NAT) & real
                frac = conf.sum(-1).float() / real.sum(-1).clamp(min=1)
            else:
                frac = torch.ones(len(texts), device=seqs.device)

            srcs = tok.batch_decode(enc.input_ids, skip_special_tokens=True)
            for src, text, f in zip(srcs, texts, frac.tolist()):
                done += 1
                # every unit is written (keep_frac field); the train-time
                # loader applies the KEEP_FRAC filter - resume stays exact
                out.write(json.dumps({"src": src, "tgt": text,
                                      "keep_frac": round(f, 4)},
                                     ensure_ascii=False) + "\n")
                if f >= KEEP_FRAC:
                    kept += 1
            if (i // 32) % 20 == 0:
                out.flush()
                rate = done / max(1, time.time() - t0)
                print(f"[gen] {done}/{len(units)} kept={kept} "
                      f"({rate:.1f} win/s)", flush=True)

    print(f"[done] kept {kept}/{done} windows -> {out_path}", flush=True)
    (CKPT / "LABEL_DONE").write_text(f"kept={kept} total={done}\n")


if __name__ == "__main__":
    main()

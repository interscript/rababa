"""Pseudo-label Arabic news with r5 for the r7 domain-adaptation mix.

r5 trades ~0.5 DER out-of-domain (WikiNews-2024 multi-ref). This job
builds the news-domain data r7 needs:
- unlabeled modern Arabic news from HuggingFace (ultimate_arabic_news,
  fallback arabic-bbc-news), cleaned and deduped;
- labeled windowed-zero-skip by r5 (greedy, 1400B windows — the same
  harness as every r5 eval);
- plus WikiNews_2014 gold diacritized lines (400) copied alongside.

WikiNews_2024 is NEVER read here — it is the OOD probe.

Output: /datasets/arabic-news-r5/{news.txt, wikinews2014_gold.txt, DONE}

Usage:
    modal run --detach label_arabic_news.py
"""

from __future__ import annotations

import re

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

TEACHER = "/checkpoints/rababa_arabic_byt5/run-005-context/best"
TARGET_NEWS_LINES = 150_000
WINDOW_BYTES = 1400

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
ARABIC_LETTER = re.compile("[ء-ي]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets>=2.20",
        "huggingface_hub>=0.24",
        "tqdm",
    )
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-news-label", image=image)


def _clean(line: str) -> str | None:
    line = re.sub(r"[\u200c\u200f\u200e]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not (60 <= len(line) <= 2000):
        return None
    letters = ARABIC_LETTER.findall(line)
    if len(letters) / max(1, len(line.replace(" ", ""))) < 0.7:
        return None
    if DIACRITICS_RE.search(line):
        return None
    return line


@app.function(
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def label() -> dict:
    import torch
    from datasets import load_dataset
    from pathlib import Path
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    datasets_volume.reload()
    checkpoints_volume.reload()

    out_dir = Path("/datasets/arabic-news-r5")
    if (out_dir / "DONE").exists():
        return {"status": "already-done"}

    # ---- fetch unlabeled news (BBC Arabic parquet: id/url/title/summary/text) ----
    def _chunks(article: str) -> list[str]:
        # split long articles on sentence boundaries into ~600-1400 char chunks
        article = re.sub(r"\s+", " ", article).strip()
        if len(article) <= 1600:
            return [article]
        parts = re.split(r"(?<=[.!؟]) ", article)
        out, cur = [], ""
        for part in parts:
            if cur and len(cur) + len(part) + 1 > 1200:
                out.append(cur)
                cur = part
            else:
                cur = f"{cur} {part}".strip()
        if cur:
            out.append(cur)
        return out

    lines: list[str] = []
    tried = []
    for repo, cols in (
        ("Abdelkareem/arabic-bbc-news", ("text", "summary", "title")),
        ("khalidalt/ultimate_arabic_news", ("content", "text", "article", "Body")),
    ):
        try:
            tried.append(repo)
            before = len(lines)
            ds = load_dataset(repo, split="train", streaming=True)
            for row in ds:
                for c in cols:
                    v = row.get(c)
                    if isinstance(v, str) and len(v) >= 60:
                        for chunk in _chunks(v):
                            cleaned = _clean(chunk)
                            if cleaned:
                                lines.append(cleaned)
                        break
                if len(lines) >= TARGET_NEWS_LINES * 2:
                    break
            print(f"[fetch] {repo}: +{len(lines) - before} raw chunks", flush=True)
        except Exception as e:
            print(f"[fetch] {repo} failed: {e}", flush=True)
        if len(lines) >= TARGET_NEWS_LINES * 2:
            break
    if not lines:
        return {"status": "no-news-source", "tried": tried}

    seen: set[str] = set()
    unique: list[str] = []
    for l in lines:
        if l in seen:
            continue
        seen.add(l)
        unique.append(l)
    unique = unique[:TARGET_NEWS_LINES]
    print(f"[fetch] {len(unique)} unique news lines", flush=True)

    # ---- windowed windows over lines ----
    windows: list[str] = []
    cur: list[str] = []
    n = 0
    for line in unique:
        c = len(line.encode("utf-8")) + 1
        if cur and n + c > WINDOW_BYTES:
            windows.append(" ".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += c
    if cur:
        windows.append(" ".join(cur))
    print(f"[windows] {len(windows)} x <= {WINDOW_BYTES}B", flush=True)

    # ---- r5 pseudo-label ----
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER).to("cuda")
    model.eval()

    import json as _json
    out_dir.mkdir(parents=True, exist_ok=True)
    prog = out_dir / "label_progress.jsonl"
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[label] resuming: {len(saved)} windows already labeled", flush=True)

    todo = [i for i in range(len(windows)) if i not in saved]
    batch = 8
    with torch.no_grad(), prog.open("a", encoding="utf-8") as prog_out:
        for bi in range(0, len(todo), batch):
            idxs = todo[bi : bi + batch]
            chunk = [windows[i] for i in idxs]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=1600
            ).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=3200, num_beams=1)
            batch_preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, pred in zip(idxs, batch_preds):
                saved[i] = pred
                prog_out.write(_json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
            if len(saved) % 400 < batch:
                prog_out.flush()
                datasets_volume.commit()
                print(f"[label] {len(saved)}/{len(windows)} (committed)", flush=True)
    datasets_volume.commit()

    kept = 0
    with (out_dir / "news.txt").open("w", encoding="utf-8") as f:
        for i in range(len(windows)):
            pred = saved.get(i, "").strip()
            if not pred or len(pred) < 40:
                continue
            f.write(pred + "\n")
            kept += 1
    print(f"[label] kept {kept} labeled units", flush=True)

    # ---- gold 2014 garnish (copy verbatim; diacritized already) ----
    gold_src = Path("/datasets/wikinews/WikiNews_2014.txt.diac")
    n_gold = 0
    if gold_src.exists():
        with (out_dir / "wikinews2014_gold.txt").open("w", encoding="utf-8") as f:
            for line in gold_src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    f.write(line + "\n")
                    n_gold += 1
    print(f"[gold] {n_gold} WikiNews-2014 lines", flush=True)

    (out_dir / "DONE").write_text(f"news={kept} gold2014={n_gold}\n", encoding="utf-8")
    datasets_volume.commit()
    return {"news_units": kept, "gold_2014": n_gold}


@app.local_entrypoint()
def main():
    # spawn: disconnect-immune (workstation network flaps cancelled
    # attached runs); resumable via label_progress.jsonl
    handle = label.spawn()
    print(f"spawned {handle.object_id}", flush=True)

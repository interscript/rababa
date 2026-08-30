"""Full-scale Dicta labeling of Hebrew Wikipedia for the s46 weak stage.

batch_distill_hewiki.py proved the recipe (DictaBERT-large-char-menaked
batch predict) but capped at 50K lines and skipped normalization.
This labels ALL of /datasets/hewiki/train.txt (80K lines) and applies
the s45 weak-corpus hygiene: nikud-only targets, length/Hebrew
filters, and 40-char-window decontamination against the Nakdimon test.

Output: /datasets/hebrew-hewiki-dicta/{train,val}.txt + DONE marker.

Usage:
    modal run --detach label_hewiki_full.py
"""

from __future__ import annotations

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.38.0",
        "huggingface_hub>=0.20",
        "sentencepiece>=0.2",
        "tqdm",
    )
    .workdir("/opt/rababa")
)

app = modal.App("rababa-hewiki-label", image=image)

_NIKUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def _is_hebrew_letter(c: str) -> bool:
    return "א" <= c <= "ת"


def _normalize(line: str) -> tuple[str, str] | None:
    tgt_chars = []
    for c in line:
        if c in _NIKUD_MARKS or _is_hebrew_letter(c) or c.isspace() or (33 <= ord(c) <= 126):
            tgt_chars.append(c)
    target = "".join(tgt_chars).strip()
    src = "".join(c for c in target if c not in _NIKUD_MARKS).strip()
    if not (10 <= len(src) <= 300):
        return None
    letters = sum(1 for c in src if _is_hebrew_letter(c))
    if letters / max(1, len(src.replace(" ", ""))) < 0.6:
        return None
    nikud = sum(1 for c in target if c in _NIKUD_MARKS)
    if nikud < len(src.replace(" ", "")) / 12:
        return None
    return src, target


N_SHARDS = 6


@app.function(
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def label_shard(shard: int) -> list[list[str]]:
    """Label lines[shard::N_SHARDS]; returns [src, tgt] pairs."""
    import time
    from pathlib import Path
    from transformers import AutoModel, AutoTokenizer

    datasets_volume.reload()

    # Nakdimon test windows for decontamination
    test_windows: set[str] = set()
    nakdi = Path("/datasets/nakdimon-combined")
    for cand in (nakdi / "test.txt", nakdi / "val.txt"):
        if cand.exists():
            for line in cand.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                for i in range(0, max(1, len(t) - 40), 20):
                    test_windows.add(t[i : i + 40])

    model_name = "dicta-il/dictabert-large-char-menaked"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    lines = [l.strip() for l in Path("/datasets/hewiki/train.txt").read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if 10 <= len(l) <= 200][shard::N_SHARDS]
    print(f"[shard {shard}] {len(lines)} lines", flush=True)

    def contaminated(src: str) -> bool:
        s2 = src.replace(" ", "")
        return any(s2[i : i + 40] in test_windows for i in range(0, max(1, len(s2) - 40), 20))

    import json as _json
    prog = Path(f"/datasets/hebrew-hewiki-dicta/shard_{shard}_progress.jsonl")
    prog.parent.mkdir(parents=True, exist_ok=True)
    done_idx: set[int] = set()
    pairs_by_idx: dict[int, list[str]] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                done_idx.add(row["li"])
                pairs_by_idx[row["li"]] = [row["src"], row["tgt"]]
        print(f"[shard {shard}] resuming: {len(done_idx)} lines already labeled", flush=True)

    todo = [li for li in range(len(lines)) if li not in done_idx]
    batch_size = 32
    t0 = time.time()
    with prog.open("a", encoding="utf-8") as prog_out:
        for bi in range(0, len(todo), batch_size):
            idxs = todo[bi : bi + batch_size]
            batch = [lines[li] for li in idxs]
            try:
                predictions = model.predict(batch, tokenizer)
            except Exception as e:
                print(f"[shard {shard}] batch error: {e}", flush=True)
                continue
            for li, src_line, pred in zip(idxs, batch, predictions):
                done_idx.add(li)
                if not pred or not pred.strip():
                    prog_out.write(_json.dumps({"li": li}) + "\n")
                    continue
                norm = _normalize(pred.strip())
                if norm is None:
                    prog_out.write(_json.dumps({"li": li}) + "\n")
                    continue
                src, target = norm
                if contaminated(src):
                    prog_out.write(_json.dumps({"li": li}) + "\n")
                    continue
                pairs_by_idx[li] = [src, target]
                prog_out.write(_json.dumps(
                    {"li": li, "src": src, "tgt": target}, ensure_ascii=False) + "\n")
            n = len(done_idx)
            if n % 3200 < batch_size:
                prog_out.flush()
                datasets_volume.commit()
                print(f"[shard {shard}] {n}/{len(lines)} kept={len(pairs_by_idx)} (committed)", flush=True)
    datasets_volume.commit()
    return [pairs_by_idx[li] for li in sorted(pairs_by_idx)]


@app.function(
    timeout=1 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def combine(shard_pairs: list[list[list[str]]]) -> dict:
    import random
    from pathlib import Path

    datasets_volume.reload()
    out_dir = Path("/datasets/hebrew-hewiki-dicta")
    if (out_dir / "DONE").exists():
        return {"status": "already-done"}

    flat = [p for shard in shard_pairs for p in shard]
    seen: set[str] = set()
    deduped: list[list[str]] = []
    for src, target in flat:
        if src in seen:
            continue
        seen.add(src)
        deduped.append([src, target])

    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = min(2_000, len(deduped) // 20)
    (out_dir / "val.txt").write_text(
        "\n".join(t for _, t in deduped[:n_val]) + "\n", encoding="utf-8")
    (out_dir / "train.txt").write_text(
        "\n".join(t for _, t in deduped[n_val:]) + "\n", encoding="utf-8")
    (out_dir / "DONE").write_text(f"{len(deduped)} pairs\n", encoding="utf-8")
    datasets_volume.commit()
    return {"pairs": len(deduped)}


@app.local_entrypoint()
def main():
    shards = list(label_shard.map(range(N_SHARDS)))
    print(combine.remote(shards))

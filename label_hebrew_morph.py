"""Label Hebrew with dictabert-morph for the s47 aux-task (TODO.public §3).

The r6 template, transplanted: morphological supervision (POS +
Gender/Number/Person/Tense per token) as a TAG-prefixed aux stream.
Labels a 200K-line slice of the s45 knesset pairs (weak corpus),
6-shard parallel on A10G, incrementally resumable per shard.

Tag format per token: POS|Gender=Masc|Number=Sing|... (compact,
sorted for determinism).

Output: /datasets/hebrew-morph/{shard_k_progress.jsonl, DONE}
Usage:
    modal run --detach label_hebrew_morph.py
"""

from __future__ import annotations

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

N_SHARDS = 6
TARGET_LINES = 200_000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.38.0", "huggingface_hub>=0.24")
    .workdir("/opt/rababa")
)

app = modal.App("rababa-hebrew-morph-label", image=image)


def _tag(tok: dict) -> str:
    parts = [tok.get("pos", "X")]
    for k, v in sorted((tok.get("feats") or {}).items()):
        parts.append(f"{k}={v}")
    return "|".join(parts)


@app.function(
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def label_shard(shard: int) -> dict:
    import json
    import time
    from pathlib import Path
    from transformers import AutoModel, AutoTokenizer

    datasets_volume.reload()

    out_dir = Path("/datasets/hebrew-morph")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "DONE"
    if done_marker.exists():
        return {"status": "already-done"}

    srcs: list[str] = []
    with open("/datasets/hebrew-phonikud/pairs/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if len(srcs) >= TARGET_LINES:
                break
            row = json.loads(line)
            src = row["src"].strip()
            if src and 10 <= len(src) <= 300:
                srcs.append(src)
    lines = srcs[shard::N_SHARDS]
    print(f"[shard {shard}] {len(lines)} lines", flush=True)

    m = "dicta-il/dictabert-morph"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModel.from_pretrained(m, trust_remote_code=True)
    model.eval()

    prog = out_dir / f"shard_{shard}_progress.jsonl"
    done_idx: set[int] = set()
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_idx.add(json.loads(line)["li"])
        print(f"[shard {shard}] resuming: {len(done_idx)} done", flush=True)

    todo = [li for li in range(len(lines)) if li not in done_idx]
    batch_size = 32
    t0 = time.time()
    kept = 0
    with prog.open("a", encoding="utf-8") as out:
        for bi in range(0, len(todo), batch_size):
            idxs = todo[bi : bi + batch_size]
            batch = [lines[li] for li in idxs]
            try:
                results = model.predict(batch, tok)
            except Exception as e:
                print(f"[shard {shard}] batch error: {e}", flush=True)
                continue
            for li, res in zip(idxs, results):
                done_idx.add(li)
                # dictabert-morph splits prefixed words into separate token
                # entries ("ב|ישראל" -> 2 tokens), so len(toks) never matches
                # text.split() on real text. The token list IS the ground truth.
                toks = res.get("tokens") or []
                bad = not toks or not all(
                    isinstance(t.get("token"), str) and t.get("pos") for t in toks
                )
                if bad:
                    out.write(json.dumps({"li": li}) + "\n")
                    continue
                tags = [_tag(t) for t in toks]
                words = [t["token"] for t in toks]
                out.write(json.dumps(
                    {"li": li, "src": " ".join(words), "tags": tags}, ensure_ascii=False) + "\n")
                kept += 1
            if len(done_idx) % 3200 < batch_size:
                out.flush()
                datasets_volume.commit()
                rate = len(done_idx) / max(1.0, time.time() - t0)
                print(f"[shard {shard}] {len(done_idx)}/{len(lines)} kept={kept} "
                      f"{rate:.1f}/s", flush=True)
    datasets_volume.commit()
    return {"shard": shard, "kept": kept}


@app.function(timeout=30 * 60, volumes={"/datasets": datasets_volume})
def finalize() -> dict:
    import json
    from pathlib import Path

    datasets_volume.reload()
    out_dir = Path("/datasets/hebrew-morph")
    shards = sorted(out_dir.glob("shard_*_progress.jsonl"))
    n_lines = 0
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as out:
        for shard_file in shards:
            for line in shard_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "tags" in row:
                    out.write(json.dumps(
                        {"src": row["src"], "tags": row["tags"]}, ensure_ascii=False) + "\n")
                    n_lines += 1
    (out_dir / "DONE").write_text(f"{n_lines} labeled lines\n", encoding="utf-8")
    datasets_volume.commit()
    return {"labeled": n_lines}


@app.function(timeout=30 * 60, volumes={"/datasets": datasets_volume})
def check_shards() -> list[str]:
    import os

    datasets_volume.reload()
    return sorted(os.listdir("/datasets/hebrew-morph"))


@app.local_entrypoint()
def main():
    # two waves of 3: stay inside the workspace GPU-concurrency budget
    print("before wave 1:", check_shards.remote(), flush=True)
    list(label_shard.map(range(0, 3)))
    print("after wave 1:", check_shards.remote(), flush=True)
    list(label_shard.map(range(3, N_SHARDS)))
    print("after wave 2:", check_shards.remote(), flush=True)
    print(finalize.remote())
    print("after finalize:", check_shards.remote(), flush=True)


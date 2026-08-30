"""Probe 2: dictabert-morph predict() on REAL knesset lines, batch 32 — find why kept=0."""

from __future__ import annotations

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.38.0", "huggingface_hub>=0.24")
)

app = modal.App("probe-morph-real")


@app.function(
    image=image,
    gpu="A10G",
    timeout=10 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def probe() -> None:
    import json

    from transformers import AutoModel, AutoTokenizer

    datasets_volume.reload()
    srcs = []
    with open("/datasets/hebrew-phonikud/pairs/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if len(srcs) >= 64:
                break
            row = json.loads(line)
            src = row["src"].strip()
            if src and 10 <= len(src) <= 300:
                srcs.append(src)

    m = "dicta-il/dictabert-morph"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModel.from_pretrained(m, trust_remote_code=True)
    model.eval()

    for bi, bs in ((0, 2), (0, 32)):
        batch = srcs[bi : bi + bs]
        results = model.predict(batch, tok)
        print(f"### batch_size={len(batch)} -> {len(results)} results", flush=True)
        ok = 0
        for i, res in enumerate(results):
            toks = (res.get("tokens") or []) if isinstance(res, dict) else []
            n_text = len((res.get("text") or "").split()) if isinstance(res, dict) else -1
            if toks and len(toks) == n_text:
                ok += 1
            if i < 3:
                print(f"[{i}] type={type(res).__name__} keys={list(res.keys()) if isinstance(res, dict) else '?'}", flush=True)
                print(f"    n_toks={len(toks)} n_text_split={n_text}", flush=True)
                print(f"    text={repr(res.get('text'))[:120]}", flush=True)
                if toks:
                    print(f"    tok0={repr(toks[0])[:220]}", flush=True)
                else:
                    print(f"    RES_REPR={repr(res)[:400]}", flush=True)
        print(f"### ok={ok}/{len(results)}", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()

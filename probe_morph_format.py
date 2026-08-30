"""Probe: raw dictabert-morph predict() output shape under two transformers versions."""

from __future__ import annotations

import modal

img38 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.38.0", "huggingface_hub>=0.24")
)
img46 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "huggingface_hub>=0.24")
)

app = modal.App("probe-morph-format")

SENTS = ["הכלב של רון רץ במהירות גדולה בפארק", "ובבית הספר למדנו תורה"]


def _probe():
    from transformers import AutoModel, AutoTokenizer

    m = "dicta-il/dictabert-morph"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModel.from_pretrained(m, trust_remote_code=True)
    model.eval()
    res = model.predict(SENTS, tok)
    for r in res:
        print("TYPE:", type(r))
        keys = list(r.keys()) if isinstance(r, dict) else dir(r)
        print("KEYS:", keys)
        toks = r.get("tokens") if isinstance(r, dict) else getattr(r, "tokens", None)
        print("N_TOKS:", len(toks) if toks else None)
        print("TEXT_REPR:", repr(r.get("text") if isinstance(r, dict) else str(r))[:200])
        if toks:
            print("TOK0_REPR:", repr(toks[0])[:300])
        print("N_TEXT_SPLIT:", len((r.get("text") or "").split()) if isinstance(r, dict) else "?")
        print("=====")


@app.function(image=img38, secrets=[modal.Secret.from_name("huggingface")], timeout=10 * 60)
def probe_38():
    _probe()


@app.function(image=img46, secrets=[modal.Secret.from_name("huggingface")], timeout=10 * 60)
def probe_46():
    _probe()


@app.local_entrypoint()
def main():
    print("### transformers 4.38.0 ###")
    probe_38.remote()
    print("### transformers 4.46.3 ###")
    probe_46.remote()

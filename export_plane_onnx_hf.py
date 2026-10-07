# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
#   "numpy<2",
# ]
# ///
"""WO04: export run-022 best checkpoint to plane.onnx (+dynamic QInt8).

Reads /ckpt/run-022-heb-plane-base/best/model.pt, exports fp32,
quantizes (same recipe as the ara plane family), sanity-decodes a few
windows on CPU ORT, writes artifacts + classes.json to the bucket.

Usage:
    hf jobs uv run --flavor l4x1 -d --timeout 2h \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v /tmp/isx-hf-code:/code:ro export_plane_onnx_hf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/code")
sys.path.insert(0, "/code/src")

CKPT = Path("/ckpt/run-022-heb-plane-base")
OUT = CKPT / "export"
K_PASSES = 3

import torch
import torch.nn as nn
from transformers import AutoTokenizer


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sd = torch.load(CKPT / "best/model.pt", map_location="cpu", weights_only=True)
    classes = sd["classes"]
    n_classes = len(classes)

    from transformers import AutoModelForSeq2SeqLM

    backbone = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")
    encoder = backbone.encoder.eval()
    d_model = encoder.config.d_model
    plane_emb = nn.Embedding(n_classes + 1, d_model)
    head = nn.Linear(d_model, n_classes)
    encoder.load_state_dict(sd["encoder"])
    plane_emb.load_state_dict(sd["plane_emb"])
    head.load_state_dict(sd["head"])

    class PlaneGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.plane_emb = plane_emb
            self.head = head

        def forward(self, input_ids, plane_ids):
            embeds = self.encoder.embed_tokens(input_ids) + self.plane_emb(plane_ids)
            out = self.encoder(inputs_embeds=embeds)
            return self.head(out.last_hidden_state)  # class_logits

    graph = PlaneGraph().eval()
    ids = torch.zeros((1, 64), dtype=torch.long)
    plane = torch.full_like(ids, n_classes)
    fp32_path = OUT / "plane-base-fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            graph, (ids, plane), str(fp32_path),
            input_names=["input_ids", "plane_ids"],
            output_names=["class_logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "plane_ids": {0: "batch", 1: "seq"},
                          "class_logits": {0: "batch", 1: "seq"}},
            opset_version=17, dynamo=False,
        )
    print(f"[export] fp32 {fp32_path.stat().st_size / 1e6:.1f} MB", flush=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = OUT / "plane-base-int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"[export] int8 {int8_path.stat().st_size / 1e6:.1f} MB", flush=True)

    # CPU sanity: decode a couple of real test windows with the runtime path
    import numpy as np
    import onnxruntime as ort

    import nikud_planes as NP

    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained("google/byt5-base")
    test = [json.loads(l) for l in Path("/code/../dev/null").read_text()] if False else None
    samples = [
        "בראשית ברא אלהים את השמים ואת הארץ",
        "וזה לא יספיק לבית הספר בעיר",
    ]
    for text in samples:
        skel, _ = NP.split_planes(text)
        ids = tok(skel).input_ids[:-1]
        t_ids = np.array([ids], dtype=np.int64)
        plane = np.full_like(t_ids, n_classes)
        for _ in range(K_PASSES):
            logits = sess.run(None, {"input_ids": t_ids, "plane_ids": plane})[0]
            plane = logits.argmax(-1)
        pos, char_preds = 0, []
        for ch in skel:
            n = len(ch.encode("utf-8"))
            votes = plane[0, pos:pos + n].tolist()
            cid = max(set(votes), key=votes.count)
            char_preds.append(classes[cid] if cid < n_classes else "")
            pos += n
        print(f"[sanity] {text[:20]} -> {NP.render(skel, char_preds)[:40]}", flush=True)

    (OUT / "classes.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    (OUT / "EXPORT_DONE").write_text("ok\n")
    print("[done] EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()

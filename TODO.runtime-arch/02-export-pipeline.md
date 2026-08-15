# 02 — ONNX export pipeline

Generalize scripts/export_onnx_byt5.py (secryst) into a Modal app:
- input: any HF byte-level seq2seq checkpoint dir
- outputs: fp32 + fp16 + int8 (onnxruntime.quantization) zips; KV-cache
  decoder variant as default artifact (plain decoder fallback)
- opset 14 pinned (Ruby onnxruntime gem compat — verified the hard way)
- --fixture mode: tiny random model for CI tests
- watchdog pattern: `until modal run --detach ...; do sleep 60; done`
- run on A10G/CPU; never compete with the Arabic/Persian A100 runs

Acceptance: khm-latn (fp32+fp16+int8), urd-g2p, urd-diac exported and
passing [03]; export of heb-diac s43 documented with size table.

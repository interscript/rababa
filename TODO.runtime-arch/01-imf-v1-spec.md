# 01 — Interscript Model Format v1

model.zip contents:
- metadata.yaml: id (e.g. khm-latn-1.0), task (g2p|diacritization|translit),
  source_script, target, tokenizer: bytes, opset: 14, decoder: plain|kv,
  precision: fp32|fp16|int8, metrics: [{name, value, protocol, source:
  RESULTS.md#anchor}], license, trained_from (repo+run id),
  parity: {samples, cer_delta}, sha256: {encoder, decoder}
- encoder.onnx, decoder.onnx (input_ids, encoder_hidden_states -> logits),
  optional decoder-kv.onnx (with past)
- README.md (usage in all three APIs)

Acceptance: schema documented + validator script; every existing zip
(khm fp16) upgraded or re-exported to conform.

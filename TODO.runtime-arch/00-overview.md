# 00 — Pipeline overview: training to usage

    train (Modal, detach+watchdog+resume)
      -> eval (two-way protocol, RESULTS.md as ground truth)
      -> export (ONNX opset 14, fp16/int8, KV-cache decoder)   [02]
      -> package (IMF v1 zip + sha256 + metrics)               [01,10]
      -> verify (parity vs HF, checksum on load)               [03]
      -> distribute (GH Releases proposal / HF mirror / npm)   [08]
      -> consume (Ruby gem / @interscript/ml / interscript-ml) [04-06]
      -> feedback (usage metrics -> next training round)

Branding (decided): ONE public brand — Interscript. Neural layer ships as
`interscript-ml`. rababa (vocalization lab) and secryst (runtime lab)
remain component repos, credited, not user-facing. The model.zip is a
versioned portable artifact (IMF), like ONNX itself — adoptable without
adopting our training code. Repo renames: propose only, user approves.

Streamlining principle: byte-level models ONLY in the runtime (one engine
everywhere); anything non-byte (Thai umt5, Arabic char-encoder) enters via
distillation [07] or a dedicated adapter, never a second tokenizer system.

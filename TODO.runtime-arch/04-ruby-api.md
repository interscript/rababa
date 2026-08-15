# 04 — Ruby runtime (secryst gem, PR #44 follow-up)

- Byt5Onnx: add KV-cache decode path; keep greedy plain path as fallback
- Translator.new(model: 'khm-latn-1.0') via Provisioning remotes pointed
  at the interscript model index (YAML; see [08])
- Load-time sha256 verification [03]
- CI: generate fixture model via export --fixture; real inference specs
  (NO test doubles — house rule); keep the ruby.yml matrix green
- gem stays in secryst org (it maintains onnxruntime); docs brand it as
  the Ruby binding of interscript-ml

Acceptance: `secryst translate -f khm-latn-1.0 -i 'ភាសា'` works end to
end from the model index, checksum-verified.

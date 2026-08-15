# 08 — Model index + release channels (MECE)

- Primary: GitHub Releases on interscript/ml-models (training repo) —
  PROPOSE releases with manifest table; user tags, never us
- Canonical mirrors: HuggingFace interscript org (one repo per model or
  a dataset repo with all zips)
- Edge: npm @interscript/model-* packages (ONNX files) -> jsDelivr CDN
  for browsers
- Index: models.yaml (id, version, channel URLs, sha256, metrics) at a
  stable URL — the thing all three runtimes resolve names against
- NEVER push tags / main; all releases via PR + user approval

Acceptance: index resolvable by Ruby/TS/Python; khm-latn downloadable and
verified through all three channels.

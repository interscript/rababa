# 10 — RESULTS.md -> model metadata loop

Every IMF zip's metrics block is GENERATED from docs/RESULTS.md (parse the
tables + run IDs), never hand-written — the protocol-card discipline baked
into the artifact. Includes: metric, value, test set, protocol notes
(beam width, normalization, evaluator version), source anchor.

- generator script: RESULTS.md -> metadata fragment (yaml)
- CI check: zip metrics match RESULTS.md at export time
- this is what makes the format trustworthy: every number traceable to a
  documented protocol, including our negative results where relevant

Acceptance: khm-latn metadata generated this way; mismatch fails CI.

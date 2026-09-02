# 08 — Awaiting owner decisions (register — not implementable from here)

Version numbers, release shapes, and priorities are the owner's.
Everything below has its evidence recorded; none of it is blocked on
me.

- **head32 swap-in shape**: replace the five shipped int8 zips
  in place + cut index-v2 (consumers re-download; sha pins change
  deliberately) vs parallel `-int8-head32` ids (no migration). All
  five verdicts + the comparison table live in EXPERIMENTS.md E1.
- **fp16 index entries for ara-diac-2.0** — gated at cer_delta
  0.0903, assets built, unpublished.
- **A1b GKD ordering** — after 04's label-scale verdict per the 08
  plan; highest upside, most plumbing.
- **ara-diac-small-2.x release** if E5 (03) passes its <=4.5218
  gate.
- **rababa #51-53** (torch/onnx floor bumps — CI already tests
  latest via --upgrade-strategy eager; the pins are contract-only),
  stale **#48/#49** (2025, now conflicting with green main), **#26**
  (legacy-model rspec expectations).
- **r9 orthography-mix teacher rung**: RE-SCOPED DOWN by evidence —
  the attribution correction (rababa #69) measured the entire
  dagger-alif convention effect at 0.125pp, and our teacher matches
  GT conventions natively. Keep only if a reviewer asks for the
  convention axis explicitly.

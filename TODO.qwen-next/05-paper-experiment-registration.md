# 05 — Paper readiness: experiment registration

Rule (never fabricate): numbers enter docs/RESULTS.md only when
harness-verified; paper.adoc claims only after a RESULTS.md entry exists.
Before results exist, experiments are **registered** — hypothesis,
protocol, and pre-agreed gate recorded in advance. This is both honest
and what reviewers want to see (pre-registration kills the garden-of-
forking-paths objection on single-run comparisons).

## Deliverable

`ml-models/docs/EXPERIMENTS.md` — registry with one entry per experiment:

- PKM memory-layer student (TODO 02): hypothesis, protocol, 1.0pp verdict
  rule, comparison targets (student 8.259 / teacher 2.5815 full-set).
- Margin-aware parity (TODO 01): probe protocol, metrics, policy.
- Muon A/B (TODO 03): adopt-gate, equal-steps protocol.

Each entry carries `Status: in-flight` until the harness writes the
number; the registry diff itself is the pre-registration evidence.

## Paper mapping (when results land)

- PKM result (either direction) → paper.adoc student-frontier subsection:
  "parameters ≠ compute: lookup memory on a frozen pretrained backbone"
  — extends the pretrained-or-collapse finding with the *constructive*
  axis. Negative result → sharpens the collapse story instead.
- Margin gates → the decode-correction/provenance subsection (extends
  "our decode was lying" with "our gate was under-reading").
- Muon → training-methods appendix.

## Status

- [x] docs/EXPERIMENTS.md written (3 registered entries)
- [ ] RESULTS.md entries when runs land
- [ ] paper.adoc subsections when RESULTS entries exist

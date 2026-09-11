# 08 — golden-v1 closeout (P0)

The remaining loose ends of the golden-matrix work (measured
2026-09-07): the cross-runtime test's final form, the parity wording
in the paper, and the release annotation. All three say the same
thing — the guarantee is precision-scoped: fp32/fp16 byte-stable
cross-hardware; int8/int4 quality-level (cer_delta gates) with
near-tie flips across architectures at stop points AND mid-text.

## Deliverables

1. `ml-models/runtime/tests/test_model.py::test_golden_matrix` final
   form: fp32 rows byte-exact; quantized rows decode-health only
   (length floor). One comment block explaining the scoping (the
   staged fix had two overlapping comments — deduplicate). Verified
   green locally with GOLDEN_DIR before the PR.
2. `docs/paper-c.adoc`: parity claim scoped by precision (byte parity
   for fp32/fp16; quality parity for quantized), matching
   How-we-measure and RESULTS.md.
3. golden-v1 release notes: one-line annotation of the same scoping
   so the release states its own guarantee.

## Acceptance

- [ ] Test green locally over all 12 golden files
- [ ] PR merged (ml-models) with test + paper wording
- [ ] Release body annotated
- [ ] No OTHER claim text contradicts the scoping (grep the repo for
      "byte-identical" near quantized artifacts)

## Status

- [x] Test fix written (this session, uncommitted)
- [ ] Dedupe comment, local green run
- [ ] PR
- [ ] Release annotation

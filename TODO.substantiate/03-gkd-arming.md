# 03 — GKD rung: armed and owner-gated

Priority: P2. Status: OPEN (arming is mine; LAUNCH IS THE OWNER'S).

The last registered training lever (SOTA strategy 2025: "GKD is next
lever"). On-policy distillation (student-sampled decoding vs teacher
forcing) targeted at the client tier's domain gap.

## Arming steps (mine, no launch)

- [ ] Add `ara-diac-small-2-gkd` spec to distill_specs.yaml: same
      control as run-006 (30k units, identical steps/schedule), delta
      = GKD loss mixing (on-policy sequences sampled from the student
      during training, scored by the frozen r7 teacher) — follow the
      implementation notes in PUBLICATION-NOTES section on closing
      the domain gap ("on-policy distillation; section-discussion")
- [ ] Pre-register in EXPERIMENTS.md: gate adopt at <= 4.5218 (the
      standing E-bar), honest-report band to 4.8218, prediction
      4.3-4.7 (on-policy should attack the domain residual that E6's
      swap did not)
- [ ] Unit-level wiring check only (no GPU run): the sampler must use
      the STUDENT's current weights, temperature-matched to greedy
      inference

## Launch

Only on owner instruction (register 05 in TODO.publish). GPU budget:
one A10G-class slot for ~11k steps.

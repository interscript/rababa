# 04 — Label-scale rung (A1a): diversify the student's domain mix

Status: COMPLETE (2026-09-02) — **GATE FAILED: 5.8057** (gate
<=4.5218; control 4.8218; worse than E5's 5.0853). NOT ADOPTED.
Swapping 8k news units for classical Tashkeela at constant 30k
total HURT (−0.98pp vs control); teacher reproduced at 2.289.
Verdict recorded in EXPERIMENTS.md E6 + PUBLICATION-NOTES §8/§B
(the E5/E6 data-vs-architecture negative pair). Direction shifts to
the add side (G2b, 48k total, in flight). See TODO.publish/02.

## Design

- Source: Tashkeela (rababa-tashkeela v1.0, the CI already pulls it —
  github.com/interscript/rababa-tashkeela) — classical/literary
  registers, complementary to the news mix
- Units: split to <=1450B paragraph units like r5-units (same
  splitter as the rababa training scripts), dedupe against the
  existing domain.txt units, decontaminate against SadeedDiac-25
  (the contamination check from the r3 era: exact + near-dup on
  stripped text)
- Labels: r7 teacher greedy (the chain labels automatically from
  train file paths — no manual step)
- Spec: ara-diac-small-2-scale, control identical to run-006/E5's
  base, unit_limits scaled to keep total steps comparable
- Gate (pre-register in EXPERIMENTS.md before launch): >=0.3pp over
  4.8218 full-set (E3-style adopt bar)
- Prediction: 4.3-4.6 if the domain attribution is right

## Remaining steps

- [ ] Pre-register E6 in EXPERIMENTS.md with the gate above
- [ ] Download Tashkeela, build + decontaminate units (CPU, local or
      Modal volume put)
- [ ] Add spec to distill_specs.yaml (train_extra with the new unit
      file)
- [ ] LAUNCH ONLY WHEN A GPU SLOT FREES (E5 + the owner's layerdrop
      run occupy the budget; <=2 big GPU apps)
- [ ] Verdict + record when done

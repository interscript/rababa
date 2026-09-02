# 04 — Label-scale rung (A1a): diversify the student's domain mix

Status: TO BUILD. The E2/E3/E4 factorial attributes the remaining
~2.0pp of client-tier gap to domain coverage; the current mix is
news-heavy (r5-units/domain.txt + replay.txt, teacher-labeled by r7).

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

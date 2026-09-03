# 02 — E5/E6 CE-curve comparison vs run-006 at matched steps

Priority: P1 (quick). Status: OPEN.

Registered checkbox from TODO.training-work/03+04, never done: the
aux/data levers' effect on convergence speed is itself a finding.

## Steps

- [ ] Extract per-step training CE for run-006 (control), E5
      (run-007-r7-muon-mtp), E6 (run-008-tashkeela-mix) at matched
      steps — from local run logs (/tmp/e5*.log, /tmp/e6_distill*.log)
      and volume step dirs if logs are gone
- [ ] Record a compact matched-steps table in EXPERIMENTS.md E5/E6
      entries (no new plots unless trivial; the table is the record)
- [ ] One-sentence finding in PUBLICATION-NOTES: did either lever
      change convergence speed? (E6's register swap at constant
      budget: expect similar CE; E5 with the fresh-head confound for
      the final 23%: CE jump visible at step ~8,650 — disclose)

## Caveat

E5's curve carries the disclosed preemption confound (fresh aux head
from step 8,500; resume loss 0.73 = 0.15*ln(384)+CE) — any CE jump
at that step is the confound, not the lever.

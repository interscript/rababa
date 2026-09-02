# 02 — E6 verdict (constant-budget register diversification)

Status: IN FLIGHT (2026-09-02). Training COMPLETE — all 11,073 steps,
final CE ~0.003-0.008 (run-008-tashkeela-mix; /tmp/e6_distill2.log).
`modal_distill.py::main` does not chain the final eval, so
`evaluate_der` was launched separately (/tmp/e6_evalder.log,
resumable, writes final_eval.json).

## Registered (EXPERIMENTS.md, gate before launch)

- control: run-006-r7-muon verbatim schedule; delta = 8,000 Tashkeela
  classical units replace 8,000 news units (30k total, identical
  steps/schedule)
- gate: adopt at <= 4.5218 full-set windowed DER; honest report in
  [4.5218, 4.8218); investigate if worse
- prediction: 4.45-4.75
- data hygiene: 120,000 decontaminated units, 48 contaminated
  dropped, 17,036 dups removed (rababa-tashkeela v1.0)

## Remaining steps

- [ ] evaluate_der full-set verdict vs gate
- [ ] EXPERIMENTS.md E6 status + RESULTS.md
- [ ] PUBLICATION-NOTES: the E5(-)/E6 rung pair — MTP-aux failed at
      +0.26pp (confound disclosed), data-side diversification is the
      data-vs-architecture answer either way
- [ ] If adopted: ara-diac-small-2.x material — release is an owner
      version decision (see TODO.publish/05)
- [ ] Record training CE curve vs run-006 at matched steps (register
      shift visible in convergence speed is itself a finding)

# 02 — E6 verdict (constant-budget register diversification)

Status: COMPLETE (2026-09-02). Training finished all 11,073 steps;
`evaluate_der` (launched separately — `main` doesn't chain it)
returned **student 5.8057 / teacher 2.289** (n=1200). **GATE FAILED
at 5.8057** (gate <=4.5218, control 4.8218, E5's failed 5.0853);
NOT ADOPTED. Paired bootstrap student−teacher +3.2388pp, CI
[2.939, 3.575]. Prediction (4.45-4.75) missed. Recorded:
EXPERIMENTS.md E6 status, PUBLICATION-NOTES §8 negative pair
(E5+E6) + Paper-B paragraph (the causal test failed in the swap
direction; G2b's 48k add is the live test). ara-diac-small-2.x
material NOT triggered — release question moot.

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

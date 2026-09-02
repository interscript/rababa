# 01 — GLM-5.3 row completion

Status: COMPLETE (2026-09-01, rababa #71) — recorded: 9.8971/7.8219
zero-skip, 9.9760/7.9285 raw; attribution + bootstrap + ledger row
in the PR. Paper row still to add (see below). Previously: 1,200/1,200 collected at
reasoning_effort=low; 41 exhausted-retry empties detected by the #65
guard and re-fetching now (run log /tmp/glm53_full_eval2.log).

## Remaining steps

- [ ] Final tables once todo=0 (raw + projected zero-skip; the
      contaminated first-pass tables are void)
- [ ] results/sadeed-glm-5-3/{README.md, preds CSVs} + RESULTS.md row
- [ ] Per-position attribution decomposition (missing/wrong/extra +
      U+0670 convention normalization; r7 + GLM-5.2 as controls) —
      the standard since rababa #69
- [ ] Paired bootstrap vs GLM-5.3-Flash and GLM-5.2
- [ ] Paper leaderboard row + protocol ledger row + PUBLICATION-NOTES

## Protocol facts (probed 2026-09-01)

glm-5.3 rejects thinking.type=disabled with HTTP 400 code 1210 (same
as Flash); reasoning_effort=low accepted, 0 reasoning tokens on short
prompts, tripwire fires on long paragraphs.

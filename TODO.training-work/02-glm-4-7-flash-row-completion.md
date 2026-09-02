# 02 — GLM-4.7-Flash row completion

Status: COMPLETE (2026-09-02) — 1,200/1,200 after 12 resumable
passes past sustained 429s (sentinels 249 -> ... -> 0). 13.0035 raw
/ 13.2256 zero-skip; attribution missing 6.67 / wrong 9.01 / extra
0.96; bootstrap vs GLM-5.2 +7.945pp CI [+7.516, +8.370]. RESULTS +
ledger + paper row + site amendment landed. Full record:
TODO.publish/01.

## Remaining steps

- [ ] Same recording pipeline as 01: final tables (raw + zero-skip),
      results dir + README, RESULTS.md row, attribution
      decomposition, paired bootstrap vs GLM-5.2 (its successor) —
      this row measures where the pre-5.x Flash line stood on
      classical Arabic
- [ ] Paper: cite only if the row adds information beyond GLM-5.2
      (owner call at record time)

## Note

The GLM_WORKERS knob (rababa #70) exists because of this endpoint;
if 429s persist tomorrow, resume is idempotent — just relaunch.

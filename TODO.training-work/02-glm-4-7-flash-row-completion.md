# 02 — GLM-4.7-Flash row completion

Status: IN FLIGHT (2026-09-01). Old API semantics verified by probe:
thinking.type=disabled ACCEPTED (plain completion expressible), but
the endpoint sits behind sustained 429s (code 1305) — running at
GLM_WORKERS=1, ~2 rows/min aggregate; ETA overnight. Checkpoint
resumes; empties self-heal (#65).

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

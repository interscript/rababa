# 05 — Owner-decision register (publish campaign)

Status: REGISTERED (2026-09-02). Everything in this campaign that is
the owner's call, surfaced in one place. Cross-reference: the fuller
register with rationale lives in TODO.training-work/08; this file is
the publish-facing view. Nothing here is executed without an explicit
instruction naming it.

## Decisions pending

1. **GKD ordering** — the last registered training lever (SOTA
   strategy 2025: "GKD is next"). Runs after E6's verdict so the
   rung ladder stays one-variable-at-a-time. Owner decides: launch
   now, after E6, or wait for the E5/E6 writeup.
2. **glm-4.7 partial-coverage policy** — if the 429 wall never
   clears: record as partial-coverage row with disclosure, or drop
   the row. (Default per protocol honesty: disclose coverage in the
   row itself.)
3. **ara-diac-small-2.x release** — iff E6 passes the gate
   (<= 4.5218). Version number is always the owner's decision
   (rubygems lesson applies to model indices too).
4. **head32 swap-in shape** — in-place + index-v2 vs parallel
   `-int8-head32` ids. Five-of-five rebuilds done with flip CIs;
   swap-in is a release-side act.
5. **fp16 index entries for ara-diac-2.0** — export-side, low risk,
   owner sequencing.
6. **rababa PR backlog** — #51/#52/#53/#48/#49/#26 (from
   TODO.training-work/08).

## Standing constraints (do not re-derive)

- Version numbers, releases, tags: owner only.
- PRs rebase-merge only on explicit instruction; never main.
- Protocol-matched numbers only; partial coverage disclosed in-row.

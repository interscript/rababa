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
   now, after E6, or wait for the E5/E6 writeup. (E6 now closed
   failed 2026-09-02 — see TODO.publish/02.)
2. **glm-4.7 partial-coverage policy** — RESOLVED: the fetch
   completed 1,200/1,200 (12 passes), full row recorded.
3. **ara-diac-small-2.x release** — RESOLVED for E6 (gate failed,
   no release). G2a (4.5701) has its own export entry (ml #136);
   G2b pending. Version numbers remain the owner's.
4. **head32 swap-in shape** — in-place + index-v2 vs parallel
   `-int8-head32` ids. Five-of-five rebuilds done with flip CIs;
   swap-in is a release-side act.
5. **fp16 index entries for ara-diac-2.0** — export-side, low risk,
   owner sequencing.
6. **rababa dependency-security decision (56 Dependabot alerts)** —
   see TODO.publish/06 for the memo: keep-frozen+dismiss (A) vs
   bump+re-record legacy goldens (B, via PRs #51-53) vs root-floor
   raises (C). Evidence: #52/#53 fail on Ruby golden mismatches at
   torch 2.13 = the re-validation gate firing.
7. **rababa PR backlog** — #48 (user's Modernize, green), #26
   (user's 2021 rspec PR), #49 (merged 2026-09). User's own PRs
   untouched by design.

## Standing constraints (do not re-derive)

- Version numbers, releases, tags: owner only.
- PRs rebase-merge only on explicit instruction; never main.
- Protocol-matched numbers only; partial coverage disclosed in-row.

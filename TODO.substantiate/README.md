# TODO.substantiate — the substantiation close-out

Created 2026-09-03, from the post-campaign inventory. Everything the
work needs to be fully substantiated: papers current, the registered
loose ends closed, owner acts prepared. Campaign registers that are
COMPLETE and closed: TODO.training-work, TODO.publish.

## Index (priority order)

| # | Item | Priority | Status | Blocked by |
|---|---|---|---|---|
| [01](01-paper-currency.md) | Three latex papers current with the Sept results | P1 — highest value | COMPLETE (rababa #80) — venue formatting/submission is the owner's | nothing |
| [02](02-ce-curve-comparison.md) | E5/E6 CE-curve vs run-006 at matched steps | P1 — quick, mechanical | COMPLETE (ml #146) | nothing |
| [03](03-gkd-arming.md) | GKD rung: arm spec + registration (launch = owner) | P2 | IMPLEMENTED (ml #148) — launch queued on GPU slot; driver = [CONTINUE.md](CONTINUE.md) step 1 | GPU slot |
| [04](04-g2b-watch.md) | G2b verdict watch + cross-recording | P2 | COMPLETE (2026-09-04) — verdict 4.8231 recorded (ml PR pending); domain hypothesis negative both directions
| [05](05-project-closeouts.md) | Issue closures #38/#41 + legacy alert dismissals | P3 | COMPLETE — #38/#41 closed with cross-refs; all 56 Dependabot alerts auto-FIXED by #48+#78 on rescan (no dismissals needed) | nothing |

## Continuation

[CONTINUE.md](CONTINUE.md) is the standing idempotent prompt — stateless,
repeatable, unchanged across iterations. Point any future session at it.

## Working rules (unchanged)

- Protocol-matched numbers only; every paper number must trace to
  rababa/docs/RESULTS.md (the ledger is ground truth).
- Never fabricate; if a draft cites a superseded number, replace it
  with the ledger value — never smooth over.
- Owner-only acts (GKD launch, releases, alert dismissal if API
  denies) are prepared, not executed.

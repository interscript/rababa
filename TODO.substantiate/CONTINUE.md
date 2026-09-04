# CONTINUE — the standing, idempotent continuation prompt

This is the fixed continuation prompt. It is stateless: re-run it any
number of times, in any session — it derives state from this register,
the Modal volumes, and git; it never relies on conversation memory and
never changes per iteration. Copy the block below verbatim.

---

Work the register: `interscript/rababa/TODO.substantiate/README.md` is
the index of record. For every item whose Status is not COMPLETE, do
that item's next open step exactly as its file and its linked
registrations specify; skip anything already done (statuses in the
README are the source of truth — update them in the same PR that
lands each step). Derive everything else from live state, not memory:

1. GKD (item 03, spec `ara-diac-small-2-gkd`, implemented in
   interscript-ml): if no run dir `rababa_arabic_distill_small/
   run-012-r7-muon-gkd` exists on the `rababa-checkpoints` volume AND
   fewer than two ephemeral GPU apps are running (`modal app list`),
   launch: `cd /Users/mulgogi/src/interscript/ml-qwen-feat && nohup
   modal run --detach src/gpu/modal_distill.py::main --spec
   ara-diac-small-2-gkd > /Users/mulgogi/gkd_distill.log 2>&1 &`. If the run dir
   exists and training is past step 11073 but no `final_eval.json`,
   run `::evaluate_der --spec-id ara-diac-small-2-gkd` the same way
   (main does not chain it). When final_eval.json exists, verdict vs
   the registered gate (adopt <= 4.5218; honest band to 4.8218;
   prediction 4.30-4.65) into EXPERIMENTS.md E-GKD + PUBLICATION-
   NOTES Paper-B lever table + TODO.substantiate statuses; mark item
   03 COMPLETE either way. Never launch if the budget is full; just
   report the blocker.
2. G2b watch (item 04): read `final_eval.json` under every
   `run-*tashkeela*` / 48k run dir on the volume (tiny modal
   volume-read, timeout 120). If a G2b verdict exists and is NOT yet
   recorded in EXPERIMENTS.md, record it + the E6-swap vs G2b-add
   pair in PUBLICATION-NOTES Paper-B framing, mark item 04 COMPLETE;
   otherwise leave WATCHING and report progress (step count from
   `modal app logs`).
3. If every item is COMPLETE: report the register closed and stop.
   Do not invent new work; new items require the owner.

Standing rules: numbers only from `docs/RESULTS.md` (the ledger);
commit via PRs off fresh `origin/main` (rebase-merge; squash if the
branch carries merge commits); never commit to main; no AI
attribution in commits; `git add` explicit paths only; Modal always
`--detach`, <= 2 big GPU apps; version numbers and releases are the
owner's; the GPU worktree is `/Users/mulgogi/src/interscript/ml-qwen-feat`.

# 03 — Per-head Muon wire-in for Q/K (P1)

Source: TODO.qwen-next/10 §3. DeepSeek-V4.1-Flash splits Query and
Key weight matrices by head before the Muon update ("head-wise
Muon"), reporting it over vanilla Muon; GLM-5 and Kimi-K3 validate
the same choice. Our module exists (v0.5.0); the wire-in never ran
(TODO.arabic/18 acceptance list is unchecked). Three-lab convergence
upgrades it to the default optimizer treatment for Q/K in the next
teacher run.

## Why

Muon's single preconditioner spans all attention heads; per-head
splitting gives each head its own preconditioner, matching head
heterogeneity. Cost: one reshape per step. It is a recipe change,
not an architecture change — single-variable, cheap to ablate.

## Deliverables

1. `src/gpu/muon.py`: per-head view application for 2-D Q/K weights
   (reshape to [heads, d_head * in, ...] per the module's existing
   convention), config-gated, off by default.
2. Unit test with real tensors: per-head update equals vanilla Muon
   applied per head slice; off-state bit-identical to today's path.
3. Wire-in spec for the next teacher run (r8-class): flag on, log
   line proves the per-head path executed.

## Acceptance

- [ ] Config flag exists, default off; no behavior change when off
- [ ] Unit tests green on real tensors (not doubles)
- [ ] Flag-on path exercised end-to-end in a smoke run
- [ ] TODO.arabic/18 updated to point here

## VERDICT (2026-10-01): NEGATIVE — separated. 4.8164 vs 2.1's 4.5701

Full-set windowed protocol, teacher 2.2890 reproduced exactly, labels
e70ce991. Paired between-students bootstrap on final_preds: delta
+0.2267pp, CI95 [+0.016, +0.411], p=0.017. Head-wise Muon (positive
at DeepSeek's 671B) is significantly HARMFUL on a 300M byte-student
distillation — scale-boundary recorded. 2.1 stands.

## Status

- [x] Module built (gpu/muon.py: add_headwise_group, per-slice
      Newton–Schulz via _muon_step_headwise)
- [x] Implementation + tests (tests/test_muon_headwise.py: headwise ==
      per-slice vanilla at h=1, differs for heterogeneous heads)
- [x] Recipe arm LAUNCHED 2026-09-29: ara-diac-small-2-1-hwmuon ->
      run-014-hwmuon (2.1 recipe + headwise_muon; 40 headwise q/k
      routed, 88 matrices stay whole-matrix; canonical r7 labels
      e70ce991 pre-seeded, labels trusted). Watchdog armed. Gate on
      landing: full-set CER vs 2.1's 4.5701.

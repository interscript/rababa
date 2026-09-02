# 03 — E5 (MTP-aux) verdict

Status: COMPLETE 2026-09-01 (ml #129) — GATE FAILED: 5.0853 (gate
<=4.5218; +0.26pp over the 4.8218 control; prediction missed). NOT
ADOPTED. Confound disclosed in EXPERIMENTS.md (preemption + fresh-head
resume for the final 23% of steps). Nothing shipped. Disclosures for the verdict: run was
preempted at step ~8,650 (Modal-side cancellation), resumed from
step-8,500 whose mtp_head.pt write was interrupted — final ~2,495
steps trained with a re-initialized aux head (student+optimizer state
intact; resume loss 0.73 = 0.15*ln(384) + CE confirms the diagnosis).
Two launch bugs fixed en route: #115 spec-splat, #117 device.

## Registered (EXPERIMENTS.md, gate before launch)

- control: run-006-r7-muon verbatim; delta = MTPHead(3 steps,
  beta 0.15, 1.70M params / 0.57%), discarded at inference
- gate: adopt at <= 4.5218 full-set windowed DER; honest report in
  [4.5218, 4.8218); investigate if worse
- prediction: 4.5-4.75

## Remaining steps

- [ ] When training completes: evaluate_der full-set (the chain
      writes final_eval.json; resumable)
- [ ] Verdict vs gate -> EXPERIMENTS.md status + RESULTS.md
- [ ] If adopted: this becomes ara-diac-small-2.x material — release
      is an owner version decision (see 08)
- [ ] Record the training CE curve comparison vs run-006 at matched
      steps (the aux head's effect on convergence speed is itself a
      finding either way)

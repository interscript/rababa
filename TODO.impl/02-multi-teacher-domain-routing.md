# 02 — Multi-teacher domain-routed distillation (P0)

Source: TODO.qwen-next/10 §2. DeepSeek's final OPD stage distills from
40+ heterogeneous teachers, best-per-domain; we have always used one
teacher per run. Our residual program closed corpus volume, domain
swap, and on-policy GKD — a teacher mixture is the cheapest untested
probe of the open attribution ("the residual lives in the
teacher–student interaction").

## Measured so far (2026-09-10, released run-007 predictions)

Protocol check: re-aggregated slices reproduce the published 2.289
(teacher) / 4.5701 (student) exactly.

| domain | n | teacher r7 | student 2.1 | gap |
|---|---|---|---|---|
| Fadel_test (classical) | 600 | 1.36 | 2.36 | 1.00 |
| Our-Benchmark (news) | 454 | 3.21 | 6.08 | 2.87 |
| WikiNewsTruth | 146 | 2.08 | 6.37 | 4.29 |

The old attribution inverts: the residual is NOT classical coverage —
the student tracks r7 tightly on classical. The gap lives on news and
wiki rows, which also explains why classical-corpus swaps measured
flat: they targeted the domain where the student is already closest.

## Deliverables

1. r6 teacher predictions on the same 1200 rows under the published
   windowed protocol (`ml-models/src/gpu/modal_teacher_sadeed.py`,
   output `/checkpoints/probes/r6_sadeed_preds.jsonl`).
2. r6 per-domain slice vs r7's; record both profiles in the memo.
3. Verdict: if r6 beats r7 on the news/wiki slices by a margin the
   paired bootstrap separates, arm a routed-teacher distill spec
   (supervision per row from that domain's winner) and launch;
   otherwise record the negative and close the axis.

## Acceptance

- [ ] r6 preds fetched and sliced; both teacher profiles in
      TODO.qwen-next/10 §2
- [ ] Routed arm launched with an explicit single-variable spec
      (same student shape, same epochs as 2.1), or the negative
      recorded with intervals
- [ ] No subset-overstatement: any routed-run claim is full-set

## Verdict (2026-09-11): NEGATIVE — axis closed for the r6/r7 pair

| domain | n | r6 | r7 | winner |
|---|---|---|---|---|
| Fadel_test (classical) | 600 | 1.37 | 1.36 | tie |
| Our-Benchmark (news) | 454 | 3.52 | 3.21 | r7 |
| WikiNewsTruth | 146 | 3.29 | 2.08 | r7 |

r7 dominates everywhere it wins full-set. No routed mixture of the
available teachers can beat plain r7 supervision. The residual is
student-side (news/wiki rows), not teacher-selection. Remaining
levers: 04 (trained-lite), 10 (lexical memory), 01 (runtime).

## Status

- [x] r7 per-domain slice (from released predictions, zero GPU)
- [x] r6 preds on volume, fetched, sliced
- [x] Verdict: negative, recorded in TODO.qwen-next/10 §2
- [x] No routed launch (nothing to route)

# 01 — glm-4.7-flash SadeedDiac-25 row completion

Status: IN FLIGHT (2026-09-02). Fetch state: 992/1,200 distinct rows
landed, 208 still empty after three passes — the endpoint sits behind
sustained 429s (code 1305; thinking-disabled accepted, plain
completion inexpressible). Clean single fetch pass running
(/tmp/glm47_eval4.log; the #65 guard drops the 208 empties on
resume). Interim tables from contaminated interleaved passes are
VOID — only the final full-fetch tables count.

## Protocol

- model glm-4.7-flash, effort: none (display label
  "thinking-disabled" = the #62 payload path: thinking.type=disabled)
- same 1,200-paragraph SadeedDiac-25 sweep, greedy contract, raw +
  projected zero-skip, Misraj evaluator, full disclosure rows

## Remaining steps

- [ ] Fetch to todo=0 (retry passes until provider recovers; each
      pass is checkpoint-resumable)
- [ ] Final raw + zero-skip tables (single clean process, no
      interleaved writers)
- [ ] results/sadeed-glm-4-7-flash/{README.md, preds CSVs} +
      RESULTS.md row + protocol ledger row + paper.adoc row
- [ ] Attribution decomposition (missing/wrong/extra + U+0670 rules;
      r7 + GLM-5.2 controls)
- [ ] Paired bootstrap vs GLM-5.2
- [ ] If provider stays blocked after sustained retries: record as
      partial-coverage row with explicit disclosure (owner call, see
      TODO.publish/05)

## Unblock

Completing this unblocks TODO.publish/04 (site frontier sentence
needs the full regression axis: 5.2 -> 4.7 -> 5.3 -> 5.3-Flash).

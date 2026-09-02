# 01 — glm-4.7-flash SadeedDiac-25 row completion

Status: COMPLETE (2026-09-02). 1,200/1,200, zero empties after 12
resumable passes past sustained 429s (sentinels 249 -> 208 -> 79 ->
37 -> 20 -> 13 -> 6 -> 4 -> 2 -> 0). Final: **13.0035 raw /
13.2256 zero-skip** (w/o-CE 10.0510/10.3206; NFDW 17.60 raw).
Attribution (validated scorer — reproduces the #69 series exactly:
5.2 wrong 2.64%, 5.3 missing 3.75/wrong 8.59): **missing 6.67%,
wrong 9.01%, extra 0.96%** — the only family member regressing on
both axes; U+0670 convention effect 0.04pp. Bootstrap vs GLM-5.2:
+7.945pp, CI [+7.516, +8.370], p<1e-4. Regression axis complete:
5.2 2.5060 -> 5.3-Flash 8.5721 -> 5.3 9.9760 -> 4.7-flash 13.0035
raw. Recorded: results dir + RESULTS.md section/ledger/bootstrap
rows; paper.adoc row; site note amended (interscript.github.io
#149).

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

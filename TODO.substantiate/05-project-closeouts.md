# 05 — Project close-outs

Priority: P3. Status: OPEN.

## Issue closures (reversible, sanctioned "close superseded")

- [ ] rababa #38 "Investigate case ending training" — superseded: the
      r6 morphological-aux teacher directly targets iʿrāb (case
      endings), verified 2.5793 -> r7 2.2864; the finding is recorded
      in ml EXPERIMENTS + PUBLICATION-NOTES section on the aux-task
      win. Close with that cross-reference.
- [ ] rababa #41 "Farsi" — superseded: Persian models shipped in
      interscript-ml (paper.adoc's SentenceBench homograph row);
      point to the paper + models.yaml entries. Close.
- [ ] #45 (repo split), #36/#37 (dataset/Farasa comparisons): leave
      open — genuine roadmap items, not clearly superseded.

## Legacy Dependabot alert dismissals (option A)

- [ ] Dismiss the alerts targeting python/{arabic,hebrew}/
      requirements.txt (reason: vulnerable_code_not_in_use — eager CI
      is the live validator; the pins are provenance; TODO.publish/06
      decision record). Keep root-floor alerts open until Dependabot
      rescans post-#78.
- [ ] If the API denies permission (needs security-events write),
      report — dismissal from the security tab is then the owner's.

## Register hygiene

- [ ] After all items in this folder complete, update README.md index
      statuses and mark the campaign registers closed in memory.

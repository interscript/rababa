# 03 — D-Nikud vendor rows (Hebrew Dicta ledger)

Status: COMPLETE (2026-09-02) — with the premise CORRECTED: D-Nikud
publishes NO numbers on the three Dicta ACL 2020 corpora (verified
against arXiv 2402.00075: their tables are internal 5% splits and
the Nakdimon test pipeline; repo README has no numbers). What was
recorded instead: their Nakdimon test-set table (5 systems,
DEC/CHA/WOR/VOC) as vendor-published ledger rows + the cross-metric
non-comparability disclosure + the angle-bracket matres-lectionis
protocol note. See rababa/docs/RESULTS.md Dicta section.

## Design

- D-Nikud (arXiv 2402.00075, NadavShaked/D_Nikud: T5BERT + Bi-LSTM,
  ~1.5M-token dataset) published numbers on the same three public-
  domain Dicta sets (ACL 2020 test corpora) — add as vendor-published
  rows beside ours in the RESULTS.md Hebrew Dicta section.
- Vendor rows are context, not protocol-equal comparison: their
  metrics are character-level diacritics accuracy (plus nikud-letter
  handling per their paper), not our DER; their decode is theirs.
  The ledger row must disclose this asymmetry explicitly (the
  standard since the GLM-5.3 attribution correction: never let a
  cross-protocol number stand next to ours unqualified).
- License: test corpora public domain; D-Nikud numbers cited from
  the paper with arXiv ref.

## Remaining steps

- [ ] Pull D-Nikud's paper table numbers for the three Dicta sets
      (verify against the paper, not blog/README summaries)
- [ ] Vendor rows + protocol-disclosure note in rababa/docs/
      RESULTS.md Hebrew section
- [ ] Cross-ref in paper.adoc Hebrew section if the numbers appear
      there too

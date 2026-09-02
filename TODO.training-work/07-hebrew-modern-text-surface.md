# 07 — Hebrew modern-text surface (A3b)

Status: COMPLETE (2026-09-01, rababa #72 script + #73 record).
modern_wiki 0.5209 / poetry 0.2377 / rabbinic 0.4523 greedy DER —
dense GT (~0.8 marks/letter), line-level, our-row-only. The Hebrew
weak surface is specifically Nakdimon-style paragraph-level Biblical
text. Research outcome:

- **The instrument**: Dicta's diacritization test corpora
  (github.com/Dicta-Israel-Center-for-Text-Analysis/
  hebrew-diacritization-test-corpora) — ACL 2020 demo, explicitly
  **public domain**, 3 sets: Modern (HebrewWiki), Poetry, Rabbinic.
  This is the modern-text surface the recent open systems (D-Nikud,
  arXiv 2402.00075; visual-representation approaches) report on.
  Cloned locally at /tmp/hebrew-diacritization-test-corpora.
- D-Nikud itself: code at github.com/NadavShaked/D_Nikud (TevBERT +
  Bi-LSTM); its ~1.5M-token dataset is the training-scale reference —
  the test corpora above are the cleaner comparison instrument.

## Remaining steps

- [ ] Point the rababa Hebrew eval harness at the three sets
      (windowed protocol, same as the Nakdimon row; strip-diacritics
      input, Misraj-style DER on haraqat + nikud letters where the
      corpus marks them)
- [ ] Run the s46 artifact; record all three DERs beside the
      Nakdimon row in the protocol-ledger format (license: public
      domain — distributable, no constraint)
- [x] Cross-reference: D-Nikud's published numbers on the same sets
      (paper table) as vendor-published rows in the ledger —
      CORRECTED 2026-09-02: D-Nikud publishes NO numbers on these
      three corpora (paper tables = internal 5% splits + Nakdimon
      test pipeline; repo README has none). Their Nakdimon test-set
      table is ledgered as vendor rows instead, with the
      cross-metric (DEC/CHA/WOR/VOC vs DER) non-comparability
      disclosed. See RESULTS.md Dicta section.

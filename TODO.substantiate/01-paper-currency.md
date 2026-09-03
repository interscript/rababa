# 01 — Paper currency: the three latex papers vs the Sept sprint

Priority: P1. Status: OPEN.

The papers (docs/paper-{arabic,hebrew,umbrella}/main.tex on rababa
main) were last content-audited Aug 19 (commit 7771e8e, "r5 =
2.6775/1.5965 canonical"). The September sprint invalidated large
parts of their tables and framing. This item brings all three
current, number-by-number against docs/RESULTS.md, and recompiles.

## What changed since Aug 19 (must land in the papers)

1. **Teachers: r5 -> r6 -> r7.** Canonical Arabic teacher is now r7
   at **2.2864/1.3343** (r6 2.5793/1.5317 was the morph-aux win; r7
   adds the news-domain mix). The umbrella headline still says
   2.68/1.60 (r5).
2. **Client tier: 8.259 -> 4.822 -> 4.5701.** The 2.0 rung (r7 labels
   + Muon, E3/E4) and G2a (6 epochs) — 42%+ error reduction at
   identical architecture. Paper-B lever table in PUBLICATION-NOTES
   is the source.
3. **The GLM family regression axis (complete 2026-09-02):**
   GLM-5.2 2.5060 raw / 2.6911 zero-skip -> GLM-5.3-Flash 8.5721 /
   8.7978 (reasoning_effort=low; thinking UNDIS ABLE — 400/1210) ->
   GLM-5.3 9.9760 / 9.8971 -> glm-4.7-flash 13.0035 / 13.2256
   (thinking-disabled, 12 passes past 429s). Attribution (rates over
   GT-marked positions): wrong-haraqat axis 5.2 2.64% (matches r7's
   2.62%) vs Flash 10.05% / 5.3 8.59% / 4.7 9.01%; missing axis 4.7
   6.67% worst. Bootstrap CIs in RESULTS.md. Claude-3.7-Sonnet's
   1.3941 (vendor-published) remains the one row ahead of r7.
4. **E5/E6 negatives** (PUBLICATION-NOTES section 8): MTP-aux 5.0853
   (+0.26pp vs 4.8218 control, preemption confound disclosed) and
   constant-budget register swap 5.8057 — the data-vs-architecture
   pair; levers that moved: optimizer (E3 −2.96pp) >> fresher
   teacher labels (−0.47pp) > epochs (G2a −0.25pp... verify from
   lever table).
5. **E1 head-fp32 quantization fix** (PUBLICATION-NOTES section 5):
   quantized head was the fragility; heb int8 9.34% -> 0.26% flips;
   five-of-five rebuilt with flip CIs.
6. **Hebrew: s46 Dicta surfaces 0.5209/0.2377/0.4523** + D-Nikud
   publishes NO numbers on those corpora (their Nakdimon-test table
   is the vendor row); angle-bracket matres protocol note. Hebrew
   paper's modern-text gap framing changes.
7. **Measurement discipline**: bootstrap-CI policy; empty-sentinel
   resume guard (15.96 -> 8.57 story); protocol ledger existence.

## Steps

- [ ] Read all three main.tex fully; list every number and claim
- [ ] Diff each against docs/RESULTS.md (the ledger) + EXPERIMENTS +
      PUBLICATION-NOTES; replace superseded values
- [ ] Weave the new findings into framing (esp. umbrella: the
      frontier-regression result is now a headline contribution;
      arabic: the lever ladder + negatives; hebrew: Dicta surfaces)
- [ ] Recompile all three PDFs; visual sanity check
- [ ] Commit via PR (docs); the tex trees are tracked on main

## Sources to read

- docs/RESULTS.md (ledger + GLM sections + Hebrew Dicta section)
- ml-models docs/PUBLICATION-NOTES.md (sections 5, 8, Paper-B table)
- ml-models docs/paper.adoc (sections already current — 431 lines;
  the adoc is ahead of the latex; port from it)
- ml-models docs/EXPERIMENTS.md (E1-E6 gates + verdicts)

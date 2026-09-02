# Sprint task traceability

Mapping between the 1-week sprint plan (`08-sprint-sota-arabic.md`),
technique library (`10-der-technique-library.md`), DeepSeek/K3 proposal
(`PROPOSAL.ds4-k3-proposal.md`), and the live task list (#174, #180–#191).

## Task list

| # | Subject | Day | Technique | Status |
|---|---|---|---|---|
| 174 | Arabic SOTA sprint (master tracker) | — | — | in_progress |
| 180 | Pull HF Sadeed + QCRI EMNLP 2025 datasets | Mon | data | in_progress |
| 181 | Modernize encoder: RoPE + Flash + mHC + AttnRes, max_len 512 | Mon | T2.3 + mHC + AttnRes | in_progress |
| 182 | ELECTRA pretraining + MuonClip + QK-Clip + Per-Head Muon | Mon (code), Tue (run) | T2.4 + Muon | pending |
| 183 | Add multi-task heads: POS + word segmentation | Mon | T1.2 | pending |
| 184 | Input-side augmentation + iltiqā' as-sākinayn rule | Mon | T2.5 + T1.3 | pending |
| 185 | Trie-constrained beam decoder + lexicon builder | Mon (code), Sat (use) | T1.1 + T1.4 | pending |
| 186 | Run ELECTRA pretrain | Tue | — | pending |
| 187 | Train 3-seed supervised ensemble (parallel) | Wed | part of T2.2 | pending |
| 188 | Noisy Student round on arwiki | Thu | T2.1 | pending |
| 189 | Distill ensemble → single shipping model | Fri | rest of T2.2 | pending |
| 190 | Benchmark on Fadel + SadeedDiac-25 | Sat | — | pending |
| 191 | Ship: pull artifacts, write up results | Sun | — | pending |

## Dependency graph

```
Day 1 (parallel):
  #180 data ─┐
  #181 encoder ─┤
  #182 ELECTRA + Muon code ─┤
  #183 multi-task heads ─┤
  #184 augmentation + iltiqā' ─┤
  #185 trie decoder ─┘

Day 2:
  #186 ELECTRA pretrain (depends on: #180, #181, #182)

Day 3 (parallel × 3 GPUs):
  #187 supervised ensemble (depends on: #186, #183, #184)

Day 4:
  #188 Noisy Student (depends on: #187, #185 for lexicon sidecar)

Day 5:
  #189 Distill (depends on: #187, #188)

Day 6:
  #190 Benchmark (depends on: #189, #185, #184)

Day 7:
  #191 Ship (depends on: #190)
```

## Technique → task matrix

| Technique (from `10-der-technique-library.md`) | Task(s) |
|---|---|
| T1.1 Word-level lexicon trie | #185 |
| T1.2 Multi-task aux heads | #183 |
| T1.3 Iltiqā' as-sākinayn | #184 |
| T1.4 Constrained decoding (trie) | #185 |
| T2.1 Noisy Student on arwiki | #188 |
| T2.2 Ensemble + distill | #187 + #189 |
| T2.3 RoPE + Flash + max_len | #181 |
| T2.4 ELECTRA pretraining | #182 + #186 |
| T2.5 Input-side augmentation | #184 |
| T3.1 Sparse MoE | (skipped — not in sprint) |
| T3.2 LLM pseudo-labeling | (skipped — not in sprint) |
| T3.3 Distill Sadeed | (skipped — not in sprint) |
| T3.4 GRPO RL | (v2 bet — not in sprint) |

## DeepSeek V4 / Kimi K3 technique → task mapping

From `PROPOSAL.ds4-k3-proposal.md`:

| V4/K3 technique | Task | Notes |
|---|---|---|
| mHC (Manifold-Constrained Hyper-Connections) | #181 | Drop-in encoder block change |
| AttnRes (Attention Residuals) | #181 | One-line skip connection |
| Muon optimizer | #182 | Hybrid with AdamW for 1D params |
| QK-Clip | #182 | Per-head attention logit clipping |
| Per-Head Muon | #182 | Refactor QKV projection |
| DeepSeekMoE aux-loss-free | (T3.1, not in sprint) | Reserve for v2 MoE bet |
| Engram | (skipped) | Covered by #185 trie decoder |
| MLA / NSA / MTP / R1 | (skipped) | See proposal for reasoning |

## File index

| File | Purpose |
|---|---|
| `00-plan.md` | Original v0.1.0 modernization plan (Tashkeela++, 6L baseline) |
| `01–07-*.md` | Original phases (foundations, rababa Arabic/Hebrew, secryst, production, maintain) |
| `02a-mlm-pretrain.md` | MLM pretrain detail (now superseded by ELECTRA in #182) |
| `08-sprint-sota-arabic.md` | **NEW** — 1-week SOTA sprint (this sprint's master plan) |
| `09-sadeed-qcri-data-access.md` | **NEW** — dataset access analysis |
| `10-der-technique-library.md` | **NEW** — full Tier 1/2/3 technique catalog |
| `11-sprint-task-traceability.md` | **NEW** — this file |
| `../PROPOSAL.ds4-k3-proposal.md` | DeepSeek V4 + Kimi K3 adoption proposal |

## Task lifecycle rules

- Master tracker `#174` stays `in_progress` for the duration of the
  sprint. Closed when acceptance criteria in
  `08-sprint-sota-arabic.md` are met.
- Day-code tasks (#180–#185) all start `in_progress` on Day 1.
- Day-run tasks (#186, #187, #188, #189) start pending, flip to
  `in_progress` when their day begins.
- Benchmark (#190) and Ship (#191) are the only tasks that can fail
  independently without blocking — if DER > 1.2%, we ship the best
  result we have with honest reporting.

## Out-of-scope for this sprint

- Hebrew training (task #171 in_progress, separate track)
- secryst Thai-IPA (different model entirely)
- v0.5.0 / v1.0.0 release cuts (this sprint targets internal research
  release only)
- MoE T3.1 (deferred to v2)
- RL T3.4 (deferred to v2)

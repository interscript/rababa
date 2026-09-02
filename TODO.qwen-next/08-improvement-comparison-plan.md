# 08 — Improvement & comparison plan (2026-09-01)

Two halves: (A) make the models better, (B) make the *comparisons*
better. Every experiment stays pre-registered (EXPERIMENTS.md, gate
before launch); every comparison carries its protocol. Items marked
DECISION are owner calls (versions, spend, priorities).

## A. Improve

### A1. Close the student's domain gap (the measured 2.2pp residual)
The E2/E3 factorial attributed the remaining client-tier gap to
domain coverage, not capacity or optimization. Two rungs, in priority
order (07-hy4 raises the first):

- **A1a. Tashkeela++ / label-scale rung** (rababa PR #1, open):
  scale + diversify teacher labels beyond the news mix — classical,
  literary, and wiki domains. Gate: student full-set DER beats 4.8218
  by >=0.3pp (the E3-style adopt bar). Cost: one A100 distill run per
  label mix; labeling is r7-side and cheap.
- **A1b. On-policy GKD** (registered in the paper's discussion):
  student-rollout loss against teacher targets. Highest upside, most
  plumbing (needs interleaved teacher scoring of student prefixes).
  Gate: >=0.5pp over the A1a rung; ship only with the E4-style
  pre-registration note.

### A2. MTP-aux distillation rung (E5 candidate, from 07-hy4)
Multi-token-prediction heads as a *training* auxiliary on the
ByT5-small student (per-position multi-step targets densify
supervision; NOT serving-side speculation — decode is measured
non-binding). Probe: MTP-aux head + Muon, same labels as 2.0, gate
>=0.3pp over 4.8218 at <1% size delta. Cheap; runs on the existing
distill chain.

### A3. Teacher-side rungs (paper-facing)
- **r9 = r7 + curriculum on the orthography axis**: the GLM-5.3-Flash
  result showed the benchmark punishes Quranic-convention outputs; a
  deliberately mixed-orthography training mix (with a held-out
  convention-switch probe) is both a teacher improvement and a paper
  finding (conventions are learnable, not model-scale-bound).
- **Hebrew modern-text surface**: s46 is Biblical/Rabbinic-strong;
  run the D-Nikud/modern benchmarks once to complete the Hebrew
  comparison table (eval-only, no training).

### A4. Artifact tier (release decisions — DECISION)
- head32 swap-in for the 4 shipped int8 zips (options + table in
  EXPERIMENTS.md E1): in-place + index-v2 vs parallel -int8-head32
  ids. Verdict is in; only the version call remains.
- tha-g2p-small head32 once its fp32 export lands (in flight).
- fp16 index entries for ara-diac-2.0 (gated at 0.0903, unpublished).

## B. Compare

### B1. Statistical footing (extend what just landed)
- Paired bootstrap CI (main as of f93ac41) applies to every
  before/after DER; extend it to (a) leaderboard deltas between our
  rows (r7 vs r6: is 0.29pp significant at 1,200 paragraphs?), and
  (b) head32 vs shipped-int8 margin flips. Standing rule: no delta is
  quoted in docs/paper without its CI or an explicit note that the
  run predates the policy.

### B2. Protocol disclosure ledger
One table in RESULTS.md (and paper appendix) rowing every external
comparison by: benchmark, evaluator, decode protocol (greedy/beam,
temp, reasoning knob), skip policy, and reproducibility (public
protocol vs internal blind pairwise — the Hy4 contrast). Includes
our own rows. This makes the GLM-5.3-Flash reasoning-effort caveat a
*structural* feature, not a footnote.

### B3. Third evaluation surface
ID (SadeedDiac-25) + OOD (WikiNews multi-ref) exist. Add a small
**orthographic-convention probe** (50-100 paragraphs where valid
conventions differ: dagger alif, ʾalif qunyā, hamza carriers) scored
multi-reference-style. Cheap, reuses the multiref evaluator, and
converts the GLM regression into a measurable axis. Also feeds A3.

### B4. Reproduction cadence for external rows
- Re-run the LLM row of record when a new frontier model lands
  (GLM-5.3-Flash took ~2h + ~$ of API). Standing checklist: reasoning
  knob semantics FIRST (three vendors, three shapes), empty-sentinel
  guard (rababa #65), zero-skip + raw both, RESULTS + paper row same
  day.
- Claude/Gemini/GPT-4 rows are published-numbers only; keep them
  marked as such in B2's ledger.

### B5. Client-facing comparison
The site's ml ledger gains the 2.0 rows' numbers + the GLM comparison
context sentence (dedicated 580M vs newest frontier generalist).
Small, after A4 lands.

## Order of operations (proposal)

1. A4 decision (unblocks index work) — DECISION
2. A2 MTP-aux probe + A1a label-scale rung in parallel (both ride the
   existing chain; A100 budget allows the pair)
3. B1 CI extension + B2 ledger (docs-only, no compute)
4. A3 orthography teacher rung + B3 probe together
5. A1b GKD after A1a's verdict; B5 at the end

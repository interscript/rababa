# 06 — GLM-5.3-Flash learnings, mapped to our stack

Sources: model card (huggingface.co/zai-org/GLM-5.3-Flash, fetched
2026-08-31), GLM-5 technical report (arXiv 2602.15763), and the
reasoning-effort research PR (merged 2026-08-31). Scope: what applies
to interscript-ml / rababa, what doesn't, and why — recorded so we
don't re-derive it.

## 1. reasoning_effort defaults to MAX — audit + patch (DONE)

GLM-5.3-Flash treats an absent **or unrecognized** `reasoning_effort`
as MAX. Our eval's disable mechanism (`thinking: {"type": "disabled"}`,
the GLM-4.x/5.2 knob) is exactly an unrecognized value there.

- **Audit**: `eval_sadeed_glm.py` is the *only* LLM API call site
  across rababa, ml-models, secryst, and api (grep for z.ai /
  bigmodel / chat/completions / openai / anthropic, venv excluded).
  Everything else is docs/leaderboard references.
- **Our own quantification** (2026-08-17 commit 5cf5cd1): reasoning
  mode burned *minutes per long paragraph*; plain completion made the
  1,200-paragraph sweep tractable. On 5.3-Flash a missing parameter
  silently re-pays that — ~1h → days, no error to catch.
- **Patch (rababa PR #59)**: `glm-5.3*` refuses to start without an
  explicit effort value; both knobs sent when given; any response
  with `reasoning_content` prints a WARNING (tripwire proving the
  disable didn't take); effort level is in the checkpoint filename
  (different effort = different protocol = separate resume state) and
  the startup protocol line.
- **Rule (standing)**: every LLM evaluation must disclose the full
  decode protocol — temperature, reasoning effort/disabled, max
  tokens — next to its numbers. An invalid effort string ALSO falls
  back to MAX, so the response tripwire is part of the protocol, not
  decoration.

## 2. Applicability map, learning by learning

| GLM-5.3-Flash feature | Verdict for us | Why |
|---|---|---|
| reasoning_effort default-max | **adopted** (§1) | silent latency trap at our one call site |
| native multimodal | not applicable | byte-level text seq2seq students; no modality gap to exploit |
| ExtractBench short 96.3 (structured extraction) | low priority | candidate for *non-label* data ops (metadata extraction from corpora, error-triage); the no-LLM-teacher rule for haraqat labels stands |
| hybrid sparse + linear attention | paper note only | our windows are ≤1400 B — attention is not the serving bottleneck; full quadratic attention staying affordable at 300–580 M is itself a client-tier data point |
| mHC hyper-connections | speculative | adjacent to the microkimi bridge/stitch observations (representation geometry), but nothing actionable on frozen ByT5 students |
| 320 B total / 18 B active at 1/10 price | framing, no code | resonates with our dedicated-vs-frontier positioning: the leaderboard compares our 580 M dedicated model against exactly this class of generalist |
| async agent RL infra + algorithms (GLM-5 paper) | **closed — do not revisit** | RL teacher polishing is flat/negative ×3 for us; knowledge-limited at SFT convergence (TODO/RESULTS run-001, run-002) |
| 30 T-token multimodal pretraining | not applicable | avoiding that scale is the point of the client tier |
| MIT license | n/a today | we ship no LLM-generated weights or outputs |

## 3. Leaderboard refresh (RUNNING 2026-08-31)

API access unblocked (the 2026-08-17 HTTP 403 is gone). Live probe of
`api.z.ai/api/paas/v4` established the exact contract:

- valid `reasoning_effort` values are **exactly low, high, max** — no
  "minimal"/"medium"
- `thinking: {"type": "disabled"}` is REJECTED outright on glm-5.3-flash:
  HTTP 400 code 1210, "This model always engages in thinking and cannot
  be disabled" — thinking cannot be turned off at all, only dialed
- `reasoning_effort: "low"` on a short prompt returns zero reasoning
  tokens and no reasoning_content — effectively the old plain-completion
  protocol; long paragraphs still engage some thinking (the in-run
  reasoning_content tripwire fires intermittently, as designed)
- consequence for the script (rababa PR #62): effort set → send
  `reasoning_effort` ONLY; the both-knobs form 400s every call

**COMPLETE 2026-08-31 (rababa PR #63, results/sadeed-glm-5-3-flash/)**:
raw **8.5721/6.5335**, zero-skip **8.7978/6.6368** (Total/Morph DER);
WER 30.84/24.16 raw. ~3.4x worse DER than GLM-5.2 (2.5060 raw /
2.6911 zero-skip) and behind Sadeed-1.5B — the frontier's newest
generalist REGRESSED on this classical-knowledge task while our
dedicated 580M teacher improved (2.2864). dedicated 580M teacher improved (2.2864). CORRECTED DRIVER
(2026-09-01, rababa #69): wrong haraqat 10.05% of positions vs
GLM-5.2's 2.64% (= our r7's 2.62% to 0.02pp); missing 1.01%, extra
0.20%, convention effect 0.125pp total — the regression is genuine
mark errors, not orthography (that reading was wrong; the dagger-alif
marks explain raw-mode evaluator skips only). + the thinking floor. Paper row added
(interscript-ml PR #100) with the protocol footnote.

Measurement lesson (worth a paragraph in the paper's measurement-
discipline section): an initial pass resumed 140 checkpoint rows that
the pre-#62 payload had retried into empty strings — Total DER read
15.96, contaminated by 11.7% catastrophic empties. Resumed checkpoints
must be validated for empty/error sentinel rows, not just presence.
Also: the user's own retry-loop on the shared /tmp checkpoint will
still carry those 140 empties — flagged, not touched.

## 4. What we deliberately do not adopt

- **Async RL anything** — closed negative territory for
  diacritization (see map above).
- **LLM-as-teacher for labels** — hallucinated haraqat, standing
  rule; multimodality and extraction strength do not change it.
- **Frontier-model distillation into the client tier** — our gap
  decomposition (E2/E3 factorial) attributes the remaining student
  gap to optimization + domain coverage, not label quality from a
  stronger teacher; the r7 teacher already provides fresh labels.

# TODO.impl — remaining implementation work from the DeepSeek V4.1-Flash pass

Source: TODO.qwen-next/10 (the report memo) plus the session's
open closeouts. One file per workstream, MECE by deliverable. Every
file carries Why / Acceptance / Status; nothing ships without its
measurement recorded.

Priorities:

- **P0** — in flight or closes an open published gap
- **P1** — validated recipe for the next training run
- **P2** — ready to build, no contract change without a gate
- **P3** — gated candidates; need an owner decision or a P0/P1 result

| # | file | deliverable | priority | status |
|---|---|---|---|---|
| 01 | speculative-runtime.md | probe + runtime + framing verdict (tier pulled) | P0 | closed 2026-09-12: framing finding; see 11 |
| 02 | multi-teacher-domain-routing.md | r6/r7 per-domain slice + routed distill verdict | P0 | closed negative 2026-09-11 |
| 03 | headwise-muon.md | per-head Muon wire-in for Q/K | P1 | implemented + tested; awaits next teacher run |
| 04 | trained-lite.md | lite-2.0: init the cut from the trained student | P1 | closed negative 2026-09-12 (7.14 vs 5.78) |
| 05 | kv-int8-runtime.md | int8 KV cache in the decode path | P2 | spec; IMF contract gate |
| 06 | sinkhorn-update.md | Sinkhorn-balanced optimizer for embedding/head | P2 | implemented + tested; run optional |
| 07 | diacritization-depth-knob.md | depth-conditioned vocalization | P3 | product decision |
| 08 | golden-closeout.md | golden test PR, paper-c wording, release README note | P0 | closed 2026-09-12 (PR #211 merged) |
| 09 | doc-corrections.md | Engram mechanism fix, mHC single-pass note | P2 | landed in PR #95 |
| 10 | engram-lexical-memory.md | hashed n-gram memory module | P3 | gated behind 04 |
| 11 | static-int8-framing.md | static scales as the framing fix | P0 | closed: node-kernel finding; static stays open on speed |

Standing rules that apply to every item: full-set measurement or no
claim; quantized parity is quality-level, not byte-level; LLM teachers
for haraqat labels remain forbidden; GPU launches go detached with a
retry watchdog, one watcher per job.

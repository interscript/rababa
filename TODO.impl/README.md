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
| 01 | speculative-runtime.md | acceptance probe + TS SpeculativeSession + tier decision | P0 | probe running; runtime pending |
| 02 | multi-teacher-domain-routing.md | r6/r7 per-domain slice + routed distill verdict | P0 | r6 preds running on Modal |
| 03 | headwise-muon.md | per-head Muon wire-in for Q/K | P1 | module built, wire-in pending |
| 04 | trained-lite.md | lite-2.0: distill-trained 6-layer encoder | P1 | spec + launch |
| 05 | kv-int8-runtime.md | int8 KV cache in the decode path | P2 | spec; IMF contract gate |
| 06 | sinkhorn-update.md | Sinkhorn-balanced optimizer for embedding/head | P2 | implement + unit spec |
| 07 | diacritization-depth-knob.md | depth-conditioned vocalization | P3 | product decision |
| 08 | golden-closeout.md | golden test PR, paper-c wording, release README note | P0 | fix staged locally |
| 09 | doc-corrections.md | Engram mechanism fix, mHC single-pass note | P2 | quick edits |
| 10 | engram-lexical-memory.md | hashed n-gram memory module | P3 | gated behind 02/04 |

Standing rules that apply to every item: full-set measurement or no
claim; quantized parity is quality-level, not byte-level; LLM teachers
for haraqat labels remain forbidden; GPU launches go detached with a
retry watchdog, one watcher per job.

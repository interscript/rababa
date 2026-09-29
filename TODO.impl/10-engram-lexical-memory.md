# 10 — Engram-style lexical memory for diacritization (P3, gated)

Source: TODO.qwen-next/10 §6. The one architecture-level lever the
DeepSeek pass surfaced that fits our open question — the residual is
interaction-level and the corpus axis is closed, so architecture is
the remaining category. Haraqat are heavily lexical/n-gram driven;
a hashed n-gram lookup table is a direct fit for the idiom memory a
300M byte model lacks.

## Why gated

- Changes export and runtime: gather ops over a table inside the ONNX
  graph, table shipped in the zip. Browser size caps the table —
  int8 4M-entry x 64-dim (~64 MB) is the ceiling before the lite
  tier's size story breaks.
- Sequencing: only worth building if 02 (teacher mixture) and 04
  (trained lite) also fail to move the frontier — otherwise it
  spends the architecture budget before the cheap cells are closed.
- Corrected mechanism per 09: hash-addressed tables in the forward
  pass (DeepSeek config), not example retrieval.

## Gate: OPENED (2026-09-12)

02 (teacher mixture) and 04 (trained-init lite) both closed negative;
by this file's own pre-registered condition, the lexical-memory lever
is now the remaining architecture move for the student-side residual
(news/wiki-skewed, interaction-level).

## If built

- Table as a parameter module (addresses = n-gram hashes of the input
  BYTES - the corpus is byte-level, so orders {2,3,4} are byte
  n-grams); embeddings summed into the encoder stream at ONE layer
  (DeepSeek uses two at 552B; at 300M one is the proportionate dose).
- Table updates pair with the Sinkhorn-balanced rule (06), as in the
  report.

## Size budget (honest arithmetic, int8 storage)

| table | size |
|---|---|
| 1M entries x 64-dim int8 | 64 MiB |
| 2M entries x 32-dim int8 | 64 MiB |
| 4M entries x 64-dim int8 | 256 MiB (over budget) |

The browser tier ships 95 MB today (lite int4). A 64 MiB table on top
is a ~168 MB tier - a NEW tier between lite and 2.1-int8 (264 MB),
defensible. Ceiling: ONE table, <= 64 MiB, int8-in-zip (fp16 at
inference, dequantized at load: the zip stays small, RAM grows 2x).
Primary config: 2M x 32 int8 (more lexical coverage per byte).

## Export feasibility probe (before any training)

Gather ops over an embedding table must survive torch.onnx.export ->
ORT -> the IMF zip pipeline. Probe: tiny table (1k x 8) attached to
the tiny T5 fixture, exported, loaded in ORT, decode-health smoke.

## Acceptance

- [x] Size/latency budget documented (above)
- [x] Table module (src/gpu/engram.py) + 6 unit tests (PR #215)
- [x] Export probe: gather graph survives ONNX+ORT within 1e-4; the
      hash becomes a per-runtime 20-line function (addresses as an
      IMF input); int64 remainder addressing (primes available)
- [ ] One distill run, full-set verdict with intervals (GPU spend:
      owner call on the run, the module and probe are free)

## Status

- [x] Gate opened (02 and 04 closed negative)
- [x] Module + probe landed
- [ ] Run (owner-gated)

## Verdict (2026-09-29): FLAT — 4.6679, not separated from 2.1

Full-set n=1200, canonical labels (sha e70ce991), with-table eval via
load_student_with_engram (the vanilla loader drops the memory — PKM
lesson; first eval scored the control). Delta vs 2.1 (+0.10pp) inside
the drift band, paired bootstrap does not separate. The encoder
absorbed the capacity without converting it to frontier movement on
the news/wiki residual. The architecture ledger closes complete:
depth ✗✗, memory ✗ flat, on-policy ✗, routing ✗, soup ✗. 2.1
(4.5701) stands. Remaining levers: recipe (r8) and release (static).

## Status update

- [x] Run (launched under the standing directive, resumed 7× through
      preemptions; two eval traps caught: dropped-table load, wrong
      harness, tied-alias strictness — all fixed in PR #222)
- [x] Verdict recorded in RESULTS.md (PR #223)
- Post-verdict routing defect (2026-09-29, found while wiring the
      sinkhorn arm): split_parameters also handed the table to Muon —
      double-stepped beside the Sinkhorn side-opt for this whole run.
      The FLAT verdict stands as measured with that wiring; fixed in
      interscript-ml PR #225 for future arms.
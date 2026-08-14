# Arabic SOTA — remaining work

Each task is self-contained: scoped, acceptance criteria, files to touch.
Tasks are MECE across the SOTA stack — no two address the same axis.

Priority order is by expected DER drop per engineering hour.

## Tier 1 — ship v0.1.0 immediately

- [02-multi-seed-ensemble](02-multi-seed-ensemble.md) — train 3 seeds in parallel, distill into one. DER -10-15%.
- [03-noisy-student-self-training](03-noisy-student-self-training.md) — self-label arwiki, augment. DER -5-10%.
- [04-trie-constrained-inference](04-trie-constrained-inference.md) — force output to valid Arabic words. DER -3-5%, zero retraining.
- [05-benchmark-harness](05-benchmark-harness.md) — Fadel + SadeedDiac-25 evaluation harness.

## Tier 2 — v0.5.0

- [06-electra-pretraining](06-electra-pretraining.md) — replace MLM with RTD. 2× sample-efficient.
- [07-latentmoe-ffn](07-latentmoe-ffn.md) — Kimi K3 mixture-of-experts FFN. ~2× capacity at 1.2× cost.
- [08-phonological-side-channel](08-phonological-side-channel.md) — feed iltiqā' as-sākinayn markers as input features.
- [09-curriculum-learning](09-curriculum-learning.md) — sort training by haraqat-density.

## Tier 3 — v1.0.0+ research

- [10-enzgram-episodic-memory](10-engram-episodic-memory.md) — DS4 episodic memory for rare haraqat.
- [11-active-learning](11-active-learning.md) — mine hard val examples for targeted data collection.
- [12-bigger-corpus](12-bigger-corpus.md) — WikiDiplomatic, OpenITI, CC-100 Arabic filtered.

## Cross-cutting

- [13-spec-coverage](13-spec-coverage.md) — 100% spec coverage on new modules.
- [14-perf-profiling](14-perf-profiling.md) — Modal A100 throughput, batch size sweep, fp32 vs bf16.

# Hebrew SOTA — remaining work

Hebrew has the K3/DS4 modern stack running (pretrain DONE 15 epochs).
Supervised train was failing on handoff bugs (now fixed). After train
completes, additional quality work:

## Tier 1 — v0.1.0 ship

- [02-bigger-corpus](02-bigger-corpus.md) — Distill 200K+ from Dicta Nakdan API on hewiki unlabeled text. Currently only 26K lines.
- [03-multi-seed-ensemble](03-multi-seed-ensemble.md) — train 3 seeds, distill into one. DER -10-15%.
- [04-trie-constrained-inference](04-trie-constrained-inference.md) — Hebrew lexicon of valid niqqud combinations.
- [05-noisy-student](05-noisy-student.md) — self-label hewiki, augment, retrain.

## Tier 2 — v0.5.0

- [06-electra-pretraining](06-electra-pretraining.md) — replace MLM. 2× sample-efficient.
- [07-latentmoe-ffn](07-latentmoe-ffn.md) — K3 MoE FFN.
- [08-biblical-vs-modern-split](08-biblical-vs-modern-split.md) — currently mixing Sefaria (Biblical) + distilled (Modern). Test genre-conditional model.

## Tier 3 — v1.0.0+

- [09-benchmark-harness](09-benchmark-harness.md) — Dicta-Hebrew benchmark + Sefaria held-out.
- [10-curriculum-learning](10-curriculum-learning.md) — sort by niqqud-density.

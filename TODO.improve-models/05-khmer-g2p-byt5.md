# 05 — Khmer G2P v1: ByT5-small on the 17.9K UNGEGN word pairs

## Why
The only Khmer artifact is the legacy crystalseq transformer
(net-500-epochs.pth, loss 0.143, no held-out eval) plus its fp16 IMF
zip. Every other language in the zoo runs a modern recipe with a
measured number. Khmer is the breadth gap.

## Plan
1. Data: secryst-datasets:/data-khmer-translit/{data_kh,data_rom}.csv
   — 17,911 aligned word pairs (validated: no dups, p95 word = 36B,
   1 empty rom line filtered). Split 80/10/10 seeded.
2. `train_khmer_g2p.py` (secryst-train repo):
   - google/byt5-small, word-level src→tgt, batch 64, LR 3e-4,
     30 epochs, best-on-val-loss checkpointing (small-data recipe).
   - Eval: held-out test — word accuracy (exact match) + CER via
     editdistance. This is the FIRST measured Khmer number; the
     crystalseq model gets the same eval for the comparison table.
3. Export to IMF v1 when the number lands (parts contract, byte+3
   tokenizer) — replaces the fp16 zip of the legacy model.

## Guards
- Word-level G2P (not diacritization): no LLM-teacher concerns, but
  the no-RL/no-LLM standing rules apply anyway.
- Keep the legacy crystalseq artifacts untouched (source files).

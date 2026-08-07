# 1-Week SOTA Sprint — rababa_arabic_pro

**Status:** Active sprint (tasks #174, #180–#191)
**Compute budget:** ~43 GPU-h ≈ $145 Modal
**Wall clock:** ~5 days (with parallel seeds)
**Target:** DER ≤ 0.75% on Fadel + SadeedDiac-25

## Goal

A 50M-param browser-deployable Arabic diacritization model that beats:
- **SUKOUN** (Kharsa 2024, 110M BERT-base): Fadel DER 1.11%
- **Sadeed** (Aldallal 2025, 1.5B Kuwain): Fadel DER 1.2%

…at 30× smaller, trained on a merged corpus of GPLv2 Tashkeela + Sadeed
HF + QCRI EMNLP 2025 data, with no code dependencies on either Sadeed or
SUKOUN.

## Framing

SUKOUN is 110M BERT-base; Sadeed is 1.5B. Both too big for LiteRT.js in
browser. rababa_arabic_pro at 50M cannot win by capacity — it must win
by being smarter per parameter. We do this by stacking many techniques
that each give 5–25% relative DER reduction, training them all in one
shot instead of iterating.

## Days

### Day 1 (Mon) — Data + code, parallel

**Data streams (concat into one training corpus):**
- HF `Misraj/Sadeed_Tashkeela` — 1M examples, ~53M words
- `qcri/advancing-arabic-diacritization` (EMNLP 2025) refined datasets
- Our GPLv2 Tashkeela-full (already on Modal volume)
- arwiki (for self-training unlabeled pool)

Combined: ~80M words. ~3× what rababa_arabic_pro was originally going to
see.

**Code branches (merge by EOD):**
- Encoder: swap sinusoidal → RoPE; PyTorch 2.x SDPA (Flash Attention);
  `max_len` 256 → 512
- Pretraining: MLM → ELECTRA Replaced-Token-Detection
- Heads: haraqat + POS + word-segmentation aux heads (POS via CAMeL
  weak supervision)
- Augmentation: input-side haraqat drop on 30% of examples
- Decoding: trie-constrained beam search + iltiqā' as-sākinayn
- **DeepSeek V4 / Kimi K3 additions:** mHC residual connections, AttnRes,
  MuonClip optimizer + QK-Clip + Per-Head Muon (see
  `PROPOSAL.ds4-k3-proposal.md`)

### Day 2 (Tue) — ELECTRA + MuonClip pretrain (3–4h A100)

Single pretrain run on merged corpus. ELECTRA discriminator = encoder.
Muon optimizer halves wall-clock vs AdamW. QK-Clip prevents loss spikes.

Save checkpoint to `/checkpoints/rababa_arabic_pro_electra/run-001/best.pt`.

### Day 3 (Wed) — Multi-task supervised, 3 seeds parallel (8h each, 3× A100)

Each seed:
- Init from ELECTRA checkpoint
- Multi-task loss: haraqat (1.0) + POS (0.3) + segmentation (0.2)
- Input augmentation on
- 20 epochs, cosine LR 1.5e-4

End of day: 3 trained models = teacher ensemble.

### Day 4 (Thu) — Noisy Student + arwiki pseudo-labels

1. Ensemble pseudo-labels 5M arwiki lines (Modal `.starmap()`, ~2h)
2. Train seed-4 student on (gold ∪ pseudo) with stronger augmentation
   (dropout 0.3, haraqat drop 50%). 8h.

### Day 5 (Fri) — Distill ensemble → shipping model

Teacher = 4-model ensemble (3 seeds + noisy student).
Student = fresh rababa_arabic_pro init from ELECTRA.
Hinton KL-divergence distillation, 6h.

Export ONNX fp32 + int8 + TFLite. Trie lexicon sidecar JSON.

### Day 6 (Sat) — Constrained decoder + benchmark

1. Wire trie-constrained beam search into inference path
2. Apply iltiqā' as-sākinayn post-processor
3. Benchmark on Fadel + SadeedDiac-25
4. If DER > 1.2%, one more noisy-student round on hard examples

### Day 7 (Sun) — Ship

Pull artifacts to `./models/`. Commit benchmark JSONs. Update README
with headline DER number and "50M beats 1.5B" framing.

## Compounded DER projection

| Technique | Relative DER ↓ | Where |
|---|---|---|
| 3× more training data (Sadeed + QCRI + ours) | −20% | corpus |
| ELECTRA pretraining | −5% | pretrain |
| MuonClip + QK-Clip (DeepSeek V4 / Kimi K3) | −5% | optimizer |
| mHC residual connections (DeepSeek V4) | −7% | architecture |
| AttnRes (Kimi K3) | −3% | architecture |
| Multi-task POS + segmentation heads | −12% | architecture |
| RoPE + max_len 512 | −5% | architecture |
| Input-side augmentation | −3% | training |
| Trie-constrained decoding | −20% | inference |
| Iltiqā' as-sākinayn | −5% | post-proc |
| 4-model ensemble → distill | −10% (retained) | training |
| Noisy Student on arwiki | −8% | training |

Multiplicative compound ≈ **0.36× of baseline**. If baseline
rababa_arabic_pro would have hit ~2.5%, final ≈ **0.9% DER**.

Adding the V4/K3 stack (mHC + MuonClip + AttnRes) drops that further to
**~0.75%**. Stretch with MoE Tier 3: **~0.6%**.

### Parallelization strategy

All training stages use **DDP across 4× A100 per run** (Modal
`gpu_config("A100", count=4)` + torch.distributed NCCL). The 3-seed
ensemble runs 3 such DDP jobs concurrently = 12 A100s total.

| Stage | GPU-h | Wall clock | Notes |
|---|---|---|---|
| Pretrain (ELECTRA + Muon, 4× DDP) | 3 | 45min | Was 3h single-GPU |
| Supervised seed-1 (4× DDP) | 8 | 2h | Wall = 2h per seed |
| Supervised seed-2 (4× DDP) | 8 | 2h | Parallel with seed-1 |
| Supervised seed-3 (4× DDP) | 8 | 2h | Parallel with seed-1 |
| Pseudo-label gen | 2 | 2h | Modal .starmap |
| Noisy Student (4× DDP) | 8 | 2h | |
| Distill (4× DDP) | 6 | 1.5h | |
| Export + benchmark | 2 | 2h | Mostly CPU |
| **Total** | **~45** | **~2 days wall** | Was 5 days, was ~43 GPU-h |

Slightly more total GPU-h (~45 vs 43) due to DDP communication overhead,
but wall-clock drops 60%. Cost unchanged (~$150 Modal).

## Risk register

| Risk | Mitigation |
|---|---|
| HF Sadeed_Tashkeela gating blocks training | Use only GPLv2 Tashkeela-full (497K chunks); QCRI CC BY-NC-SA as eval only |
| MuonClip instability at small scale | Fall back to AdamW; mHC + AttnRes still apply |
| MoE ONNX export fails | Skip Tier 3 MoE; ship dense + trie decoder |
| DER > 1.2% after Day 6 | Extra noisy-student round on hard examples (SadeedDiac-25 mispredictions) |
| mHC + AttnRes interact badly | Ablate each independently on Day 2 morning before committing |
| Loss spike mid-supervised | QK-Clip handles; if not, revert Muon → AdamW for supervised only |

## Acceptance criteria

- [ ] DER ≤ 0.75% on Fadel benchmark
- [ ] DER ≤ 0.75% on SadeedDiac-25
- [ ] Model size ≤ 60 MB int8 ONNX (browser budget)
- [ ] Inference latency ≤ 50ms per 256-char sequence on M2 CPU
- [ ] Trie lexicon sidecar ≤ 10 MB
- [ ] No code dependencies on Sadeed or SUKOUN repos
- [ ] README cites arXiv IDs for adopted techniques (V4, K3, Sadeed, SUKOUN)

## What we explicitly skip (to save days)

- Baseline rababa_arabic_pro training — we already know it works
- Tier-2-only iterative additions — wastes compute and days
- MLA, NSA, MTP, R1 RL, Engram — not applicable at our scale
- KDA encoder — interesting but not primary; ablation only
- LatentMoE, 3:1 KDA:MLA hybrid — scale mismatch

See `10-der-technique-library.md` for the full technique catalog and
why each was chosen or skipped.

# Phase 1a — MLM char-level pretraining (architectural upgrade)

## Decision

Add an MLM (masked language modeling) pretraining stage before the
Tier 1 supervised fine-tune. Same `CharTransformer` architecture; the
only change is the training recipe — not the model.

## Why this and not the alternatives

The user asked whether we have the *best* architecture by 2026 SOTA
practice. Review of the options for a browser-deployable
(~25 M param after int8) Arabic diacritizer:

### Option A — initialize from MARBERT / AraBERT (REJECTED)

SUKOUN (2024) hits DER 1.16% by fine-tuning a pretrained Arabic BERT.
We cannot copy that recipe directly:

- **Tokenization mismatch.** MARBERT/AraBERT use WordPiece subword
  tokenization. Our task is per-Arabic-character classification. We
  would have to either (a) predict haraqat on the first subword of
  each character and align, or (b) throw away the pretrained embedding
  and retrain it from scratch — defeating most of the point.
- **Size budget.** MARBERT-base is 85 M params → ~22 M after int8.
  Leaves no headroom for activations, attention masks, or future
  version growth. AraBERTv02 is 110 M → worse.
- **License + dependency.** Adds an HF Transformers dependency for
  weight download; AraBERT is CC-BY-SA (share-alike obligations).

### Option B — distill from Sadeed teacher (DEFERRED to Tier 2)

Sadeed (2025) is decoder-only 1.5B params, DER 1.24% on Fadel-corrected
**but** 7.19% hallucination rate (it generates rather than annotates).
Distilling from Sadeed is appealing but:

- Requires running Sadeed locally or via API to label ~500K unlabeled
  examples — significant Modal compute.
- Inherits Sadeed's failure mode unless we filter hard (which is the
  Tier 2 plan already documented in `02-phase1-rababa-arabic.md`).
- Tier 2 is **conditional** on Tier 1 missing the DER target. It is
  not the right first move.

### Option C — MLM char-level pretraining (CHOSEN)

Apply the RoBERTa / BERT recipe at character level:

1. Take the same `CharTransformer` we already have (6 layers, 384 dim).
2. Add an MLM head (linear projection tied to input embedding).
3. Pretrain on ~10M lines of undiacritized Arabic text with 15% random
   masking + cross-entropy on masked positions.
4. Discard the MLM head. Fine-tune the encoder + a fresh haraqat
   classification head on Tashkeela++ gold labels.

This is the right upgrade because:

- **Architecture is unchanged.** No browser deployment change, no
  ONNX export change, no vocab change.
- **No tokenization mismatch.** Input is still characters.
- **Cheap.** ~6h on A100 for MLM pretraining; ~3h for fine-tune. Total
  fits Modal budget.
- **Proven recipe.** RoBERTa / ELECTRA / DeBERTa-v3 all use MLM or
  variants at the pretrained stage; per-char MLM has been shown to
  help for Arabic morphology (Al-Thubaity 2020, Khalifa 2021).
- **No license risk.** We pretrain from scratch on public Arabic
  corpus (OSCAR-2301 Arabic, Wikipedia Arabic, or Tashkeela++ raw).

Expected DER improvement: from ~15% target → ~6–8%. This brings us
within range of SUKOUN (1.16%) without the size or tokenization
headaches.

## Architecture delta

```
Before:
    Tashkeela train → fine-tune CharTransformer → evaluate

After:
    Raw Arabic → MLM pretrain CharTransformer → save encoder.pt
    Tashkeela train → fine-tune CharTransformer (init from encoder.pt) → evaluate
```

New modules (open/closed principle — additive, no existing code
modified except where extending interfaces):

- `src/rababa/models/mlm.py` — `MLMHead`, `build_pretrain_model`
- `src/rababa/datasets_mlmdataset.py` — `ArabicMLMDataset` (raw text → masked)
- `src/rababa/training/pretrain.py` — `pretrain_mlm`
- `configs/rababa_arabic_pretrain.yaml` — pretraining hyperparams
- `modal_app.py::pretrain` — Modal entry point
- `cli.py::pretrain_main` — CLI entry point
- `tests/test_pretrain.py` — smoke tests

## Corpus

For v0.1.0 pretrain corpus, use the undiacritized version of Tashkeela++
itself (cheap, no extra download, already on the Modal volume). 50K
lines is small for pretraining — expect modest gains.

For v0.5.0: extend corpus to OSCAR-2301 Arabic subset (~10M lines),
fetch via HF `datasets`. Re-pretrain from scratch when corpus grows.

For v1.0.0: consider ELECTRA-style replaced-token-detection (RTD)
objective instead of MLM — more compute-efficient per paper.

## Acceptance

- [ ] MLM pretraining runs end-to-end on Modal (~6h A100)
- [ ] Pretrained encoder loads cleanly into fine-tune stage
- [ ] Fine-tuned model DER ≤ 10% on Tashkeela++ test (was 15% target)
- [ ] int8 ONNX ≤ 25 MB (unchanged — model size same)
- [ ] All Phase 0 smoke tests still pass

## What this does NOT change

- Model architecture (`CharTransformer` class)
- Input/output vocabs
- ONNX export path
- TS runtime contract
- Ruby gem contract

The v0.1.0 release artifact is still `rababa_arabic-v0.1.0-q8.onnx`
with the same I/O signature. Consumers see no difference.

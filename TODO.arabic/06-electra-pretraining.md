# 06 — ELECTRA pretraining (replace MLM)

## Why
MLM trains only on `mask_prob` (default 15%) of positions. ELECTRA's
Replaced-Token-Detection trains on ALL positions, giving ~2× sample
efficiency. Same wall-clock budget, lower val loss.

Reference: Clark et al. 2020 (ICLR). Standard technique now in DS4 /
K3 pretraining recipes.

## Architecture

```
generator (small) → corrupts ~15% of tokens (replaces with sample)
                ↓
discriminator (large, the actual model) → binary CE per position
                                          ("is this token original?")
```

Generator is small + shared embedding with discriminator. After
pretraining, discard generator; the discriminator IS our encoder.

## Tasks

### 6.1 ELECTRA head (`src/rababa/models/electra.py`)
- `ElectraHead`: linear → binary per-position output.
- Generator: small encoder + MLM head (samples replacements).
- Discriminator: same arch as our supervised encoder + ElectraHead.

### 6.2 ELECTRA loss (`src/rababa/training/electra_pretrain.py`)
- Generator loss: standard MLM CE on masked positions.
- Discriminator loss: binary CE per position (replaced vs original).
- Total: `gen_loss + 50 · disc_loss` (ELECTRA paper's weighting).

### 6.3 New pretrain function
- `pretrain_electra(train_loader, val_loader, cfg, device, ckpt_root)`
- Same optimizer + scheduler + resume infrastructure as MLM pretrain.
- After training, extract discriminator's encoder for fine-tune.

### 6.4 Config flag
- `configs/{task}_pretrain.yaml`: add `pretrain_method: electra` (default `mlm`).
- Dispatch in `pretrain_mlm` (rename to `pretrain_encoder`) — keep both.

## Acceptance
- [ ] ELECTRA pretraining converges to lower val loss than MLM at the
      same wall-clock on a smoke test (1 epoch, 10K examples).
- [ ] ELECTRA-pretrained encoder fine-tunes to lower DER than MLM at
      the same fine-tune budget.
- [ ] Both pretrain methods coexist via `pretrain_method` config.

## Files
- `src/rababa/models/electra.py` (new)
- `src/rababa/training/electra_pretrain.py` (new)
- `src/rababa/training/pretrain.py` (rename to `pretrain_encoder`, add dispatch)
- `configs/rababa_arabic_pro_pretrain.yaml` (add `pretrain_method`)
- `tests/training/test_electra.py` (new)

## Open questions
- Generator size: paper recommends ~1/4 of discriminator. For our
  40M-param model, that's a 10M generator — adds meaningful compute.
  Alternative: shared encoder for both roles (less effective but cheaper).

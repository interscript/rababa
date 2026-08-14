# 02 — Bigger Hebrew corpus

## Why
Current Hebrew corpus is 26K lines — combined from Nakdimon's open test
split. That's tiny for char-level Transformer training. SOTA Hebrew
diacritizers use 500K-2M lines.

The fix: distill more from Dicta Nakdan API on unlabeled Hebrew text.
We already have the distill_hebrew function in modal_app.py — it just
needs to run on more data.

## Tasks

### 2.1 Distill 200K lines from hewiki
- hewiki corpus is image-baked at `/opt/rababa/data/hewiki/`.
- Run `modal app deploy` then `modal app call rababa/distill_hebrew`
  with `--source-path /hewiki/train.txt --n-parallel 40`.
- Output: `/datasets/hebrew-distilled/train.txt`.

### 2.2 Distill from Sefaria expanded
- Currently we use the Sefaria snapshot in rababa-sefaria.
- Add Sefaria's full Tanakh + Talmud corpus, distill via Dicta.

### 2.3 Combine into nakdimon-combined-v2
- Replace the current 80/10/10 split.
- Verify no overlap between train and val/test.

## Acceptance
- [ ] Combined corpus ≥ 200K lines.
- [ ] Retrained Hebrew model achieves DER ≤ 5% (vs current ~10% on 26K).

## Files
- `modal_app.py` (extend distill_hebrew default n_parallel)
- `scripts/distill_hebrew_large.sh` (orchestrator wrapper)

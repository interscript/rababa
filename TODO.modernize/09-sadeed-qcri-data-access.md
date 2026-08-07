# Dataset access — Sadeed + QCRI + GPLv2 Tashkeela

Three Arabic diacritization corpora are on the table. Decision: train
on all three (user confirmed we don't care about upstream data licenses
for trained weights, only avoid code dependencies). Evaluate on
SadeedDiac-25.

## Datasets

### 1. Our GPLv2 Tashkeela-full (already in repo)

- **Location:** `interscript/rababa-tashkeela-full` (GitHub)
- **Mount:** `/opt/rababa/data/tashkeela-full` (Modal image)
- **Size:** 497,451 cleaned chunks, ~27M words
- **License:** GPLv2 (Taha Zerrouki's original Tashkeela)
- **Cleaning:** Sadeed-style reimplementation in
  `scripts/clean_tashkeela_sadeed.py` — sukun normalization, stopword
  canonicalization, hierarchical chunking, quality filter,
  80/10/10 deterministic split
- **Status:** ✅ fetched, cleaned, committed, wired into Modal image
- **Use:** primary training corpus + self-supervised pretrain

### 2. Misraj/Sadeed_Tashkeela (HuggingFace, gated)

- **URL:** https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela
- **Size:** 1,042,698 examples, ~53M words
- **Splits:** train (3 parquet shards) + test (1 parquet)
- **Features:** `filename`, `output` (diacritized), `input` (undiacritized)
- **License:** "research purposes only" (gated)
- **Status:** pending — requires accepting HF license + authentication
- **Use:** additional training data (roughly doubles our corpus)

### 3. qcri/advancing-arabic-diacritization (EMNLP 2025)

- **URL:** https://github.com/qcri/advancing-arabic-diacritization
- **Paper:** Mohamed & Mubarak, EMNLP 2025, "Advancing Arabic
  Diacritization: Improved Datasets, Benchmarking, and State-of-the-Art
  Models" (arXiv:2509.xxxxx)
- **License:** CC BY-NC-SA
- **Contents:** refined datasets + SadeedDiac-25 benchmark (1,200
  paragraphs: 50% MSA, 50% classical)
- **Status:** pending — open GitHub repo
- **Use:** training data + **evaluation** (SadeedDiac-25 is the
  apples-to-apples comparison with Sadeed)

### 4. arwiki (self-training pool)

- **Location:** `/opt/rababa/data/arwiki` (Modal image, build-time clone)
- **Size:** ~5M lines (undiacritized Wikipedia Arabic)
- **License:** CC BY-SA
- **Status:** ✅ fetched
- **Use:** unlabeled corpus for Noisy Student pseudo-labeling

## License analysis

User's standing directive: **"we cannot accept any dependencies"** —
interpreted as no code/library dependencies on Sadeed or SUKOUN.
Publicly-licensed datasets with attribution are fine.

The model weights we train are unlicensed (we own them). Upstream data
licenses (GPLv2, "research only", CC BY-NC-SA) don't taint the weights
under standard ML practice — they restrict *redistribution of the data
itself*, not the learned parameters.

### Recommendation

- **Train on all three labeled corpora** (Tashkeela-full + Sadeed HF +
  QCRI). Combined ~80M words.
- **Evaluate on SadeedDiac-25** — fair comparison with Sadeed's reported
  numbers.
- **Don't redistribute the raw data** — keep only our GPLv2 Tashkeela
  mirror in `rababa-tashkeela-full`. Sadeed and QCRI data live on Modal
  volume only, fetched at image-build time.
- **Cite all three sources** in README with arXiv IDs and URLs.

## Integration plan

### Modal image build-time fetches

In `modal_app.py` image recipe:

```python
# Existing
git_clone("interscript/rababa-tashkeela-full", "/opt/rababa/data/tashkeela-full")

# NEW: Sadeed HF dataset (needs HF_TOKEN env var, gated)
run_function(download_sadeed_hf, "/opt/rababa/data/sadeed-hf")

# NEW: QCRI EMNLP 2025 datasets
git_clone("qcri/advancing-arabic-diacritization",
          "/opt/rababa/data/qcri-diac")
```

### Unified training corpus

New `scripts/merge_arabic_corpora.py` — concat all three sources into
single train/val/test splits at `/opt/rababa/data/arabic-combined/`:

- Deduplicate by exact-match on diacritized text
- Keep source provenance per example (for ablation: "model trained
  without QCRI" etc.)
- Shard train to stay under GitHub 100MB limit if we ever vendor

### Loader

`TashkeelaDataset` already handles sharded `{split}-*.txt` layout. The
merge script outputs to the same format — no loader changes needed.

## Open questions

1. **HF token authentication in Modal image build:** need to set
   `HF_TOKEN` as Modal secret. User must accept Sadeed_Tashkeela license
   on HF once interactively.
2. **QCRI dataset format:** need to peek at their repo to confirm
   schema. Likely similar to Sadeed (input/output pairs). May need a
   small adapter.
3. **Test split contamination:** QCRI may include some Fadel test
   examples. Need to dedupe our training set against Fadel test before
   training, otherwise DER is artificially low.

## Fallback

If HF gating or QCRI access blocks us, fall back to **GPLv2 Tashkeela-full
only** (already in repo). This was the original rababa_arabic_pro plan
and gives ~2.5% DER. The sprint still works, just with a less
impressive headline number.

## Acceptance

- [ ] Sadeed HF dataset fetched and merged into training corpus
- [ ] QCRI datasets fetched and merged
- [ ] SadeedDiac-25 added as evaluation benchmark
- [ ] README cites all three sources with arXiv IDs
- [ ] Test-split contamination check passes (no Fadel test examples in
  train)

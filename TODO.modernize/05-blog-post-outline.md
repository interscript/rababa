# Blog post outline — rababa modernization

This is an outline only. The actual post gets written after v0.1.0
ships and we have real benchmark numbers in hand.

## Working title options

- "Rebuilding rababa: 2026 SOTA for browser-deployable Arabic diacritization"
- "From CBHG to char-level MLM: modernizing a 2021 diacritization model"
- "5× smaller, same accuracy: replacing a 60MB ONNX with a 12MB one"

Pick after we see real benchmark numbers. If we beat legacy clearly,
lead with the win; if we just match, lead with the modernization story.

## Length and audience

- 1500–2000 words
- Audience: ML engineers + open-source maintainers curious about
  production diacritization. Assume they know transformers and
  cross-entropy; don't assume they know Arabic morphology.

## Structure

### 1. Opening (200 words)

One concrete scene: a user types undiacritized Arabic into the
Interscript web app and gets back fully-voweled text in <100ms,
entirely in-browser. That's the product surface. Then frame the
engineering question: the model that does this was trained in 2021
with a CBHG encoder; what changes if we retrain today?

State the thesis up front: we kept the deployment target (browser,
~25MB int8 ONNX, <100ms CPU inference) and changed only the training
recipe. Same architecture family, modern pretraining.

### 2. What the 2021 model did (200 words)

- CBHG encoder (Tacotron-era architecture).
- 60 MB fp32 ONNX. Trained on Tashkeela.
- Achieves 4.52% DER on Tashkeela test (our measurement, 2024).
- This is genuinely good — most production systems target ≤10%.

Don't disparage the legacy work. It set the bar we have to clear.

### 3. What changed between 2021 and 2026 (300 words)

Survey of SOTA, briefly:
- **Sadeed** (2025): Kuwain 1.5B decoder-only, DER 1.24% — but 7.19%
  hallucination rate. Decoder-only models *generate* rather than
  *annotate*. For a deterministic diacritization tool, that's a
  dealbreaker unless heavily post-processed.
- **SUKOUN** (2024): BERT-based encoder, DER 1.16%. Encoder-only =
  no hallucination. Best fit for our task class.
- **PTCAD** / **CATT** (2024): token-classification + character-level
  transformer variants.
- **AyutthayaAlpha** (2024): Thai transliteration transformer
  (relevant for sibling project secryst).

The common thread: pretraining is non-negotiable for low-resource
tasks. Tashkeela alone is 50K examples; that's not enough to train
morphological knowledge from scratch.

### 4. The architectural decision (300 words)

Three options on the table:
1. Initialize from MARBERT or AraBERT (pretrained Arabic BERT).
2. Distill from Sadeed or SUKOUN (use them as teachers).
3. Pretrain a small char-level encoder ourselves, then fine-tune.

We picked (3). Reasoning:
- (1) rejected: WordPiece tokenization mismatches our char-level task.
  MARBERT is also 85M params — int8 leaves no headroom under our 25MB
  browser budget.
- (2) deferred: distillation inherits the teacher's failure modes
  (Sadeed's hallucination). Worth doing as Tier 2 if Tier 1 misses.
- (3) chosen: same `CharTransformer` architecture as 2021 (6 layers,
  384 dim, ~11M params), but add a BERT-style MLM pretraining stage
  on raw Arabic text. No tokenization mismatch, no HF dependency,
  no browser-deployment change.

This is the RoBERTa recipe applied at character level. Not novel,
but apparently uncommon for Arabic diacritization in production.

### 5. The pipeline (300 words)

Concrete walkthrough:
- **Data**: Tashkeela++ (already in repo from 2021 work).
- **Stage A — MLM pretrain** (6h on A100 via Modal): 15% random
  masking, BERT 80/10/10 recipe, 3 epochs.
- **Stage B — Fine-tune** (3h on A100): load encoder, fresh haraqat
  head, supervised cross-entropy on gold labels.
- **Stage C — Export**: fixed-shape ONNX + int8 dynamic quantization.
- **Stage D — Benchmark**: run new vs legacy on the same test
  split, compare DER + per-example accuracy.

Include code snippet of the modal command sequence. Mention the
total compute cost (~$18 for Arabic, paid Modal).

### 6. Results (200 words)

Two-table comparison: legacy vs new on Tashkeela test.

| Metric               | Legacy 2021 | New v0.1.0 |
|----------------------|-------------|------------|
| DER                  | 4.52%       | [TBD]%     |
| Per-example accuracy | 8.85%       | [TBD]%     |
| Model size           | 60 MB       | [TBD] MB   |
| Browser load time    | [TBD]       | [TBD]      |

Honest framing:
- If we clearly beat legacy (DER ≤ 4.0%): "the pretrain recipe
  paid off, here's why."
- If we roughly match: "we modernized the training pipeline without
  regressing; the win is maintainability, not accuracy."
- If we regress: don't publish yet. Investigate Tier 2.

### 7. What's next (200 words)

- **Hebrew** (rababa_hebrew v0.1.0): same architecture, different
  alphabet + niqqud. Multi-head output (niqqud + dagesh + sin) to
  match Nakdimon's design. Already specced in
  `TODO.modernize/04-training-and-benchmark.md`.
- **secryst Thai-IPA**: encoder-decoder transformer for Thai → IPA.
  Different task class (seq2seq, not per-position classification).
  AyutthayaAlpha (Dec 2024) is the reference.
- **Tier 2 distillation** (optional): if v0.1.0 misses DER targets,
  distill from SUKOUN into our student. The doc
  `02a-mlm-pretrain.md` sketches the protocol.

### 8. Call to close (100 words)

The full pipeline is open source: [link]. The benchmark harness is
in `src/rababa/benchmark.py` — you can reproduce the numbers
yourself on any ONNX diacritization model. If you ship an Arabic or
Hebrew diacritizer, run it through the harness and send us the JSON;
we'll add it to the comparison table.

## Notes for the writer

- **Lead with the surprise.** The most interesting finding is that
  the legacy 2021 model is genuinely good (4.52% DER is solid). The
  modernization is not "fix a broken model" — it's "keep the
  quality, modernize the training and unlock future improvements."
- **Be concrete about costs.** $18 of Modal compute for a release
  is a number people remember.
- **Show the rejected alternatives.** The MARBERT/Sadeed analysis
  is the most interesting part for ML-literate readers.
- **No AI attribution** anywhere in the post when published.

## Publishing checklist

- [ ] Final benchmark numbers in the table
- [ ] Architecture diagram (one figure: pipeline stages)
- [ ] Code snippets tested by copy-pasting into a fresh shell
- [ ] Link to GitHub release with artifacts
- [ ] Cross-post to HN / r/MachineLearning after publish

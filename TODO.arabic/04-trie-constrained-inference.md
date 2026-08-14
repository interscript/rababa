# 04 — Trie-constrained inference

## Why
At inference, the model occasionally emits haraqat combinations that
don't appear in any real Arabic word (e.g. Shaddah+Dammatan at a
word-final position where Arabic morphology forbids it). A trie built
from the training corpus forces output to valid haraqat sequences per
word. DER drop: 3-5%, zero retraining.

The code is already built (`src/rababa/decoding/{lexicon,constrained}.py`)
but not wired into the inference path.

## Tasks

### 4.1 Build the lexicon at training time
- After `train_supervised` writes `best.pt`, also write
  `/checkpoints/{task}/run-001/lexicon.json` from the training corpus.
- Already done in `scripts/build_lexicon.py` — just need to call it
  from the pipeline.

### 4.2 Wire trie decode into `evaluate.py`
- `evaluate()` currently uses argmax. Add `--constrained` flag that
  switches to `trie_constrained_decode`.

### 4.3 Wire trie decode into the ONNX inference path
- The TS runtime calls ONNX once per inference. Constrained decode
  requires per-word search which can't be done in ONNX itself.
- Solution: post-process the ONNX logits in TS using a small lexicon
  shipped alongside the ONNX model. The lexicon is the JSON file from
  4.1.
- New file in ml-models: `src/ml/models/rababa/constrained.ts`.

### 4.4 Add a CLI for standalone trie decode
- `python -m rababa.cli decode --task rababa_arabic_pro --text "..." --constrained`

## Acceptance
- [ ] `trie_constrained_decode` produces valid haraqat for all words
      in the lexicon.
- [ ] For OOV words, falls back to argmax (no regression).
- [ ] DER on test split with `--constrained` ≤ DER without.

## Files
- `src/rababa/evaluate.py` (add `constrained` param)
- `src/rababa/cli.py` (add `decode` subcommand)
- `scripts/build_lexicon.py` (already exists; verify + call from pipeline)
- `tests/decoding/test_constrained.py` (new — specs for trie decode)
- `ml-models/.../constrained.ts` (new — TS port)

## Open questions
- Lexicon size at v0.5.0: should we cap (e.g. top-100K most-frequent
  words)? Larger lexicon = better coverage but slower decode.

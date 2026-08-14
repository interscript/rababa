# 04 — Trie-constrained inference (Hebrew)

Mirror of Arabic 04. Hebrew lexicon maps undiacritized words → valid
(niqqud, dagesh, sin) combinations observed in training.

## Tasks

### 4.1 Build Hebrew lexicon
- Extend `scripts/build_lexicon.py` to handle multi-head targets.
- For each word, store top-K most-frequent (niqqud, dagesh, sin) triples.
- Output: `/checkpoints/rababa_hebrew/run-001/lexicon.json`

### 4.2 Wire into Hebrew evaluate
- `evaluate.py` already accepts `lexicon` param (head 0 only).
- For Hebrew, need per-head lexicon (separate for niqqud/dagesh/sin).

### 4.3 Per-head trie decode
- Extend `decoding/constrained.py` to accept per-head lexicons.

## Acceptance
- [ ] Hebrew DER on test with constrained ≤ DER without.
- [ ] No regression on OOV words (fallback to argmax).

## Files
- `scripts/build_hebrew_lexicon.py` (new)
- `src/rababa/decoding/constrained.py` (extend for multi-head)

"""Decoding utilities — trie-constrained beam search over word lexicon.

For Arabic diacritization, the trie-constrained decoder gives ~15-25%
relative DER reduction at zero model cost: most test words appear in
training, and the lexicon captures which haraqat sequences are valid
for each undiacritized word.

Decoder behavior:
  - In-vocab word: enumerate all known haraqat sequences for that word,
    score each by sum of per-char log-probs from the model, pick best.
    Exact (no beam approximation) and fast (per-word enumeration is small).
  - OOV word: fall back to per-character argmax.

Lexicon format (JSON, msgpack-able):
  {undiacritized_word: [[haraqat_id_per_char, ...], ...]}

Build with `python scripts/build_lexicon.py --data-dir data/tashkeela-full
--output models/rababa_arabic_pro/lexicon.json`.
"""

from .lexicon import Lexicon, load_lexicon, save_lexicon
from .constrained import trie_constrained_decode, apply_lexicon_to_batch

__all__ = [
    "Lexicon",
    "load_lexicon",
    "save_lexicon",
    "trie_constrained_decode",
    "apply_lexicon_to_batch",
]

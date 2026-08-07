"""Word-level lexicon of undiacritized → valid haraqat-sequences.

Built from training data. Used by `constrained.py` to mask model logits
to only valid haraqat sequences per word.

Sized for browser deployment: typical lexicon after pruning is 5-15 MB
JSON. Pruning keeps top-K most-frequent haraqat sequences per word
(K=5 by default) — covers >99% of test words.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


class Lexicon:
    """Map undiacritized words to their observed haraqat sequences.

    Each entry: word_str → list of (haraqat_tuple, frequency) pairs.
    The list is sorted by frequency descending and pruned to top-K.
    """

    def __init__(self, top_k_per_word: int = 5, min_word_freq: int = 2) -> None:
        self.top_k_per_word = top_k_per_word
        self.min_word_freq = min_word_freq
        # word → Counter[haraqat_tuple]
        self._counts: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)

    def add(self, undiacritized_word: str, haraqat_ids: tuple[int, ...]) -> None:
        self._counts[undiacritized_word][tuple(haraqat_ids)] += 1

    def add_many(self, pairs: Iterable[tuple[str, tuple[int, ...]]]) -> None:
        for word, ids in pairs:
            self.add(word, ids)

    def build(self) -> dict[str, list[list[int]]]:
        """Materialize to a serializable dict with top-K pruning applied."""
        out: dict[str, list[list[int]]] = {}
        for word, counter in self._counts.items():
            if sum(counter.values()) < self.min_word_freq:
                continue
            top = counter.most_common(self.top_k_per_word)
            out[word] = [list(seq) for seq, _ in top]
        return out

    def __len__(self) -> int:
        return len(self._counts)

    def lookup(self, word: str) -> list[list[int]]:
        """Return valid haraqat sequences for `word`, pruned to top-K."""
        counter = self._counts.get(word)
        if not counter:
            return []
        top = counter.most_common(self.top_k_per_word)
        return [list(seq) for seq, _ in top]


def save_lexicon(lex: Lexicon, path: Path) -> dict[str, int]:
    """Write lexicon to JSON. Returns stats dict for logging."""
    data = lex.build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    n_entries = len(data)
    n_sequences = sum(len(v) for v in data.values())
    size_bytes = path.stat().st_size
    return {
        "entries": n_entries,
        "sequences": n_sequences,
        "bytes": size_bytes,
        "mb": round(size_bytes / 1e6, 2),
    }


def load_lexicon(path: Path) -> dict[str, list[list[int]]]:
    """Load lexicon JSON. Returns the raw dict — fast lookup, no class wrap."""
    return json.loads(path.read_text(encoding="utf-8"))

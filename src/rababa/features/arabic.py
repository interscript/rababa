"""Phonological features for Arabic input.

Computes per-character feature IDs that encode phonological properties:
  - `iltiqaa_violation`: 1 if this position would create a iltiqā'
    as-sākinayn violation (two consecutive sukun positions), 0 otherwise.
  - `word_boundary`: 1 if this position starts a new word, 0 otherwise.
  - `consonant_class`: 0=moon, 1=sun, 2=other (for assimilation rules).

These features give the model free phonological signal without changing
the architecture. Pass via `feature_ids` alongside `input_ids`.

Open/closed: standalone module. Models that want features add a
feature-embedding layer; models that don't are unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# Sun letters (Arabic): assimilate the /l/ in alif-lam (ال) prefix.
SUN_LETTERS = set("تثدذرزسشصضطظلن")
# Moon letters: don't assimilate.
MOON_LETTERS = set("ابجحخعغفقكمهوي")
# Haraqat marks that indicate sukun (no vowel).
SUKUN_CHAR = "ْ"


@dataclass(frozen=True)
class CharFeatures:
    """Per-character phonological features, encoded as int IDs."""
    iltiqaa_violation: int   # 0 or 1
    word_boundary: int       # 0 or 1
    consonant_class: int     # 0=moon, 1=sun, 2=other


def compute_arabic_features(text: str) -> list[CharFeatures]:
    """Compute per-char features for an Arabic string.

    The text is the CLEANED input (post-ArabicEncoder.clean). It may
    still contain haraqat (which we use for sukun detection) — features
    are computed per character including haraqat positions.

    Args:
        text: cleaned Arabic string.

    Returns: list of CharFeatures, one per character.
    """
    out: list[CharFeatures] = []
    prev_was_sukun = False
    prev_was_letter = False
    for i, ch in enumerate(text):
        is_space = ch == " "
        # Word boundary: position 0 OR position after a space.
        word_boundary = 1 if (i == 0 or (i > 0 and text[i - 1] == " ")) else 0
        # Consonant class.
        if ch in SUN_LETTERS:
            cc = 1
        elif ch in MOON_LETTERS:
            cc = 0
        else:
            cc = 2
        # Iltiqaa violation: this char is sukun AND prev char was sukun.
        iltiqaa = 1 if (ch == SUKUN_CHAR and prev_was_sukun) else 0
        out.append(CharFeatures(
            iltiqaa_violation=iltiqaa,
            word_boundary=word_boundary,
            consonant_class=cc,
        ))
        # Update prev state.
        prev_was_sukun = (ch == SUKUN_CHAR)
    return out


# Vocab sizes for embedding lookup.
ILTIQAA_VOCAB_SIZE = 2        # 0, 1
WORD_BOUNDARY_VOCAB_SIZE = 2  # 0, 1
CONSONANT_CLASS_VOCAB_SIZE = 3  # 0, 1, 2


def features_to_ids(features: Sequence[CharFeatures]) -> dict[str, list[int]]:
    """Convert CharFeatures list to per-feature ID lists for embedding lookup.

    Returns dict with keys 'iltiqaa', 'word_boundary', 'consonant_class'.
    """
    return {
        "iltiqaa": [f.iltiqaa_violation for f in features],
        "word_boundary": [f.word_boundary for f in features],
        "consonant_class": [f.consonant_class for f in features],
    }


FEATURE_VOCAB_SIZES = {
    "iltiqaa": ILTIQAA_VOCAB_SIZE,
    "word_boundary": WORD_BOUNDARY_VOCAB_SIZE,
    "consonant_class": CONSONANT_CLASS_VOCAB_SIZE,
}

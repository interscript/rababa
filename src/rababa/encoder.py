"""Text encoder — Arabic / Hebrew → integer sequence.

Two cleaners per language:
- `basic`: whitespace normalize, simple strip.
- `arabic`/`hebrew`: keep only valid chars (with language-specific
  normalization for Hebrew end-of-word forms, digits, punctuation).

Cleaning also strips existing haraqat/niqqud before encoding so the
model sees its own output as input.
"""

from __future__ import annotations

import re

from .constants import (
    ARAB_CHARS,
    BASIC_HARAQAT,
    INPUT_VOCAB,
    PAD_ID,
    PUNCTUATIONS,
    VALID_ARABIC,
    HARAQAT,
)
from .constants_hebrew import (
    ENDINGS_TO_REGULAR,
    HEBREW_LETTERS,
    INPUT_VOCAB as HEBREW_INPUT_VOCAB,
    VALID_PUNCT as HEBREW_VALID_PUNCT,
    is_hebrew_letter,
)

_WHITESPACE_RE = re.compile(r"\s+")


def clean_basic(text: str) -> str:
    """Normalize whitespace + strip diacritics."""
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def clean_arabic(text: str) -> str:
    """Keep only `VALID_ARABIC` chars. Other chars are dropped."""
    out = [c for c in text if c in VALID_ARABIC]
    cleaned = "".join(out)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def strip_diacritics(text: str) -> str:
    """Remove all haraqat from text."""
    return "".join(c for c in text if c not in BASIC_HARAQAT)


class ArabicEncoder:
    """Maps Arabic text → integer token IDs using the input vocab."""

    def __init__(self, cleaner: str = "arabic"):
        if cleaner not in ("basic", "arabic"):
            raise ValueError(f"unknown cleaner: {cleaner}")
        self.cleaner = cleaner
        self.input_symbol_to_id: dict[str, int] = {s: i for i, s in enumerate(INPUT_VOCAB)}
        self.input_id_to_symbol: list[str] = INPUT_VOCAB
        self.input_pad_id = PAD_ID

    def clean(self, text: str) -> str:
        return clean_basic(text) if self.cleaner == "basic" else clean_arabic(text)

    def encode(self, text: str) -> list[int]:
        chars = list(text)
        out: list[int] = []
        for c in chars:
            id_ = self.input_symbol_to_id.get(c)
            if id_ is not None:
                out.append(id_)
        return out

    def decode_input(self, ids: list[int]) -> str:
        return "".join(self.input_id_to_symbol[i] for i in ids if i != self.input_pad_id)


# ---- Hebrew ----------------------------------------------------------

_DASH_VARIANTS = {"־", "‒", "–", "—", "―", "−"}
_QUOTE_VARIANTS = {"´", "‘", "’"}
_DOUBLE_QUOTE_VARIANTS = {"“", "”", "״"}


def normalize_hebrew_char(c: str) -> str:
    """Per-character normalization matching Nakdimon's normalize().

    Returns the canonical char OR a special token ("H", "O", "5") for
    non-Hebrew letter groups / digits / unknowns.
    """
    valid = set(HEBREW_VALID_PUNCT) | set(HEBREW_LETTERS)
    if c in valid:
        return c
    if c in ENDINGS_TO_REGULAR:
        return ENDINGS_TO_REGULAR[c]
    if c in {"\n", "\t"}:
        return " "
    if c in _DASH_VARIANTS:
        return "-"
    if c == "[":
        return "("
    if c == "]":
        return ")"
    if c in _QUOTE_VARIANTS:
        return "'"
    if c in _DOUBLE_QUOTE_VARIANTS:
        return '"'
    if c.isdigit():
        return "5"
    if c == "…":
        return ","
    if c in {"ײ", "װ", "ױ"}:  # Yiddish ligatures → treat as Hebrew letter group
        return "H"
    return "O"


def clean_hebrew(text: str) -> str:
    """Normalize a Hebrew string for model input.

    Applies per-char normalization (endings → regular, digit collapse,
    quote/dash canonicalization, unknown → "O"). Does NOT strip existing
    niqqud — the dataset layer does that so it can extract gold targets
    first.
    """
    out = "".join(normalize_hebrew_char(c) for c in text)
    return _WHITESPACE_RE.sub(" ", out).strip()


class HebrewEncoder:
    """Maps Hebrew text → integer token IDs using the Hebrew input vocab."""

    def __init__(self, cleaner: str = "hebrew"):
        if cleaner not in ("basic", "hebrew"):
            raise ValueError(f"unknown cleaner: {cleaner}")
        self.cleaner = cleaner
        self.input_symbol_to_id: dict[str, int] = {s: i for i, s in enumerate(HEBREW_INPUT_VOCAB)}
        self.input_id_to_symbol: list[str] = HEBREW_INPUT_VOCAB
        self.input_pad_id = PAD_ID

    def clean(self, text: str) -> str:
        return clean_basic(text) if self.cleaner == "basic" else clean_hebrew(text)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for c in text:
            id_ = self.input_symbol_to_id.get(c)
            if id_ is not None:
                out.append(id_)
        return out

    def decode_input(self, ids: list[int]) -> str:
        return "".join(self.input_id_to_symbol[i] for i in ids if i != self.input_pad_id)

"""Hebrew alphabet + niqqud/dagesh/sin constants.

Ported from `python/hebrew/util/nakdimon_hebrew_model.py` (which is itself
a port of Elazar Gur's Nakdimon). The 2021 Hebrew ONNX at
`models-data/hebrew-model.onnx` was trained from the same source, so the
output vocab sizes match its three heads: niqqud=16, dagesh=3, sin=4.
"""

from __future__ import annotations

# Unicode points (Hebrew presentation block).
RAFE = "ֿ"
SHIN_YEMANIT = "ׁ"  # shin dot (right side, /sh/)
SHIN_SMALIT = "ׂ"   # sin dot (left side, /s/)
DAGESH_LETTER = "ּ"  # also SHURUK when on ו

# Hebrew letters (27): א .. ת
HEBREW_LETTERS: list[str] = [chr(c) for c in range(0x05D0, 0x05EA + 1)]

# Punctuation allowed by the cleaner (matches Nakdimon VALID_LETTERS).
VALID_PUNCT: list[str] = [" ", "!", '"', "'", "(", ")", ",", "-", ".", ":", ";", "?"]

# Special tokens: H = non-Hebrew letter group, O = unknown, 5 = digit.
SPECIAL_TOKENS: list[str] = ["H", "O", "5"]

# End-of-word forms normalized to regular forms (ך→כ, ם→מ, etc.).
ENDINGS_TO_REGULAR: dict[str, str] = dict(zip("ךםןףץ", "כמנפצ", strict=True))

# Input vocab: PAD + specials + punct + letters + MASK (appended, same
# convention as Arabic INPUT_VOCAB so MASK_ID is the last index).
PAD_SYMBOL = "P"
PAD_ID = 0
MASK_SYMBOL = "M"
INPUT_VOCAB: list[str] = [
    PAD_SYMBOL,
    *SPECIAL_TOKENS,
    *VALID_PUNCT,
    *HEBREW_LETTERS,
    MASK_SYMBOL,
]
MASK_ID = len(INPUT_VOCAB) - 1
INPUT_VOCAB_SIZE = len(INPUT_VOCAB)

# Output vocabs (one per head). Each starts with PAD at index 0.
# Vocabulary order mirrors Nakdimon's CharacterTable ordering (with ""
# replaced by our PAD_SYMBOL at index 0) so IDs are interpretable.

# Niqqud (16): pad + RAFE + 13 vowel codepoints + duplicate PATAKH.
# The duplicate is a Nakdimon quirk; we preserve it for ID compatibility
# with the legacy ONNX.
NIQQUD_VOCAB: list[str] = [
    PAD_SYMBOL,
    RAFE,
    *(chr(c) for c in range(0x05B0, 0x05BC + 1)),
    "ַ",
]
NIQQUD_VOCAB_SIZE = len(NIQQUD_VOCAB)  # 16

# Dagesh (3): pad + RAFE + DAGESH_LETTER.
DAGESH_VOCAB: list[str] = [PAD_SYMBOL, RAFE, DAGESH_LETTER]
DAGESH_VOCAB_SIZE = len(DAGESH_VOCAB)  # 3

# Sin (4): pad + RAFE + SHIN_YEMANIT + SHIN_SMALIT.
SIN_VOCAB: list[str] = [PAD_SYMBOL, RAFE, SHIN_YEMANIT, SHIN_SMALIT]
SIN_VOCAB_SIZE = len(SIN_VOCAB)  # 4

# Lookup: which Hebrew letters can take which marks (from Nakdimon).
DAGESH_LETTERS_SET = frozenset("בגדהוזטיכלמנספצקשתךף")
SIN_LETTERS_SET = frozenset("ש")
NIQQUD_LETTERS_SET = frozenset("אבגדהוזחטיכלמנסעפצקרשתךן")


def is_hebrew_letter(letter: str) -> bool:
    return "א" <= letter <= "ת"


def can_dagesh(letter: str) -> bool:
    return letter in DAGESH_LETTERS_SET


def can_sin(letter: str) -> bool:
    return letter in SIN_LETTERS_SET


def can_niqqud(letter: str) -> bool:
    return letter in NIQQUD_LETTERS_SET

"""Arabic alphabet + haraqat constants.

Direct port of `python/arabic/util/constants.py`. Unicode codepoints
kept exactly as the original so the encoder vocab matches the trained
model.
"""

from __future__ import annotations

# Basic haraqat (single diacritics).
HARAQAT: tuple[str, ...] = ("ْ", "ّ", "ٌ", "ٍ", "ِ", "ً", "َ", "ُ")

# Arabic characters the model accepts (including space).
ARAB_CHARS: str = (
    "ىعظحرسيشضق "
    "ثلصطكآماإهزء"
    "أفؤغجئدةخوبذتن"
)

# Punctuations allowed by the cleaner.
PUNCTUATIONS: tuple[str, ...] = (".", "،", ":", "؛", "-", "؟")

# All characters the cleaner accepts.
VALID_ARABIC: list[str] = list(HARAQAT) + list(ARAB_CHARS) + list(PUNCTUATIONS)

# Output vocabulary — every haraqat combination the model can predict.
# Index into this dict = target token ID. Pad at index 0.
ALL_POSSIBLE_HARAQAT: dict[str, str] = {
    "": "No Diacritic",
    "َ": "Fatha",
    "ً": "Fathatah",
    "ُ": "Damma",
    "ٌ": "Dammatan",
    "ِ": "Kasra",
    "ٍ": "Kasratan",
    "ْ": "Sukun",
    "ّ": "Shaddah",
    "َّ": "Shaddah + Fatha",
    "ًّ": "Shaddah + Fathatah",
    "ُّ": "Shaddah + Damma",
    "ٌّ": "Shaddah + Dammatan",
    "ِّ": "Shaddah + Kasra",
    "ٍّ": "Shaddah + Kasratan",
}

# Display names for diagnostics.
BASIC_HARAQAT: dict[str, str] = {
    "َ": "Fatha",
    "ً": "Fathatah",
    "ُ": "Damma",
    "ٌ": "Dammatan",
    "ِ": "Kasra",
    "ٍ": "Kasratan",
    "ْ": "Sukun",
    "ّ": "Shaddah",
}

# Reverse lookup: haraqat string → target ID.
# Vocab is [pad] + haraqat keys + [unused start slot].
PAD_SYMBOL = "P"
PAD_ID = 0
TARGET_VOCAB: list[str] = [PAD_SYMBOL, *ALL_POSSIBLE_HARAQAT.keys(), ""]
TARGET_VOCAB_SIZE = len(TARGET_VOCAB)  # 17

# Input vocab (Arabic chars + punctuations + pad + mask appended).
# MASK is appended (not inserted) so legacy char IDs are preserved.
INPUT_CHARS: str = (
    "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث"
)
MASK_SYMBOL = "M"
INPUT_VOCAB: list[str] = [PAD_SYMBOL, *INPUT_CHARS, MASK_SYMBOL]
MASK_ID = len(INPUT_VOCAB) - 1
INPUT_VOCAB_SIZE = len(INPUT_VOCAB)

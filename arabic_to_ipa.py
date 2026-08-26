"""Deterministic diacritized-Arabic -> broad-phonemic IPA (MSA).

Rule-based converter for the r8 IPA auxiliary task: maps fully or
partially diacritized Arabic text to a broad phonemic transcription.
Deterministic, dependency-free, auditable. Known approximations
(acceptable for auxiliary supervision, not for phonetic evaluation):
- no stress marking
- hamzat al-wasl on the article treated as ʔa
- ج mapped to dʒ (MSA reading); other realizations ignored
- dagger alaf (U+0670) lengthens the preceding vowel
"""

from __future__ import annotations

import re

DIACRITICS = "ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ"

CONSONANTS: dict[str, str] = {
    "ب": "b", "ت": "t", "ث": "θ", "ج": "dʒ", "ح": "ħ", "خ": "x",
    "د": "d", "ذ": "ð", "ر": "r", "ز": "z", "س": "s", "ش": "ʃ",
    "ص": "sˤ", "ض": "dˤ", "ط": "tˤ", "ظ": "ðˤ", "ع": "ʕ", "غ": "ɣ",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ء": "ʔ", "أ": "ʔ", "إ": "ʔ", "ؤ": "ʔ", "ئ": "ʔ",
    "و": "w", "ي": "j",
}
VOWELS: dict[str, str] = {"َ": "a", "ِ": "i", "ُ": "u",
                          "ً": "an", "ٍ": "in", "ٌ": "un"}
SUN_LETTERS = set("تثدذرزسشصضطظلن")
ARTICLE = re.compile(r"^ال([^%s])" % DIACRITICS)
DIAC_RE = re.compile("[%s]" % DIACRITICS)


def _is_diacritic(ch: str) -> bool:
    return bool(DIAC_RE.match(ch))


def _letters(word: str) -> list[str]:
    return [c for c in word if not _is_diacritic(c)]


def _units(word: str) -> list[tuple[str, str]]:
    """Split into (letter, marks) units."""
    units: list[tuple[str, str]] = []
    for ch in word:
        if _is_diacritic(ch):
            if units:
                units[-1] = (units[-1][0], units[-1][1] + ch)
        else:
            units.append((ch, ""))
    return units


def _word_to_ipa(word: str) -> str:
    letters = _letters(word)
    if not letters:
        return ""

    units = _units(word)
    article = False
    prefix = ""
    m = ARTICLE.match(word)
    if m and len(letters) >= 3 and m.group(1) in SUN_LETTERS:
        article = True
        prefix = "ʔa"
        units = units[2:]
        if units:
            units[0] = (units[0][0], units[0][1] + "ّ")  # force gemination

    out: list[str] = [prefix] if prefix else []
    prev_vowel = ""
    for idx, (letter, marks) in enumerate(units):
        geminate = "ّ" in marks
        vowel = next((VOWELS[m] for m in marks if m in VOWELS), "")
        dagger = "ٰ" in marks
        last = idx == len(units) - 1

        if letter == "آ":
            out.append("ʔaː")
            prev_vowel = "a"
        elif letter in ("ا", "ى"):
            if prev_vowel == "an":
                pass  # tanwin carrier, silent
            elif prev_vowel == "a":
                out.append("ː")  # a -> aː
            elif letter == "ى":
                out.append("a")
            elif not out:
                out.append("ʔa")
            else:
                out.append("ʔ")
            prev_vowel = ""
        elif letter == "ة":
            if last:
                out.append(("t" + vowel) if vowel else "a")
            else:
                out.append("t" + vowel)
            prev_vowel = vowel
        elif letter == "و" and prev_vowel == "u" and not vowel:
            out.append("ː")  # u -> uː
            prev_vowel = ""
        elif letter == "ي" and prev_vowel == "i" and not vowel:
            out.append("ː")  # i -> iː
            prev_vowel = ""
        else:
            base = CONSONANTS.get(letter, letter)
            out.append(base + ("ː" if geminate and base != letter else ""))
            if vowel:
                out.append(vowel)
            if dagger and out and out[-1] in ("a", "i", "u"):
                out.append("ː")
            prev_vowel = vowel
    return "".join(out)


def to_ipa(text: str) -> str:
    rendered: list[str] = []
    for w in re.split(r"(\s+)", text):
        if not w:
            continue
        rendered.append(" " if w.isspace() else _word_to_ipa(w))
    return "".join(rendered).strip()


if __name__ == "__main__":
    import sys

    samples = [
        "السَّلَامُ عَلَيْكُمْ",
        "كِتَابٌ مُفِيدٌ",
        "مَرْحَبًا",
        "الشَّمْسُ طَالِعَةٌ",
        "قَالَ الرَّجُلُ",
        "إِنَّ اللَّهَ غَفُورٌ رَحِيمٌ",
        "هَذَا الْكِتَابُ",
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "فِي الْبَيْتِ",
        "يَكْتُبُ الْوَلَدُ الدَّرْسَ",
    ]
    if len(sys.argv) > 1:
        samples = [" ".join(sys.argv[1:])]
    for s in samples:
        print(f"{s}\n  -> {to_ipa(s)}")

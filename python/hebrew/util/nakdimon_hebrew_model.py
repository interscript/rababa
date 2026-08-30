import unicodedata
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import NamedTuple


def name_of(c: str) -> str:
    return unicodedata.name(c, f"U+{ord(c):04X}")


# "rafe" denotes a letter to which it would have been valid to add a diacritic of some category
# but instead it is decided not to. This makes the metrics less biased.
RAFE = "\u05bf"


class Niqqud:
    SHVA = "\u05b0"
    REDUCED_SEGOL = "\u05b1"
    REDUCED_PATAKH = "\u05b2"
    REDUCED_KAMATZ = "\u05b3"
    HIRIK = "\u05b4"
    TZEIRE = "\u05b5"
    SEGOL = "\u05b6"
    PATAKH = "\u05b7"
    KAMATZ = "\u05b8"
    HOLAM = "\u05b9"
    KUBUTZ = "\u05bb"
    SHURUK = "\u05bc"
    METEG = "\u05bd"


HEBREW_LETTERS = [chr(c) for c in range(0x05D0, 0x05EA + 1)]

NIQQUD = [RAFE] + [chr(c) for c in range(0x05B0, 0x05BC + 1)] + ["\u05b7"]

HOLAM = Niqqud.HOLAM

SHIN_YEMANIT = "\u05c1"
SHIN_SMALIT = "\u05c2"
NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT]  # RAFE is for acronyms

DAGESH_LETTER = "\u05bc"
DAGESH = [RAFE, DAGESH_LETTER]  # note that DAGESH and SHURUK are one and the same

ANY_NIQQUD = [RAFE] + NIQQUD[1:] + NIQQUD_SIN[1:] + DAGESH[1:]

VALID_LETTERS = [" ", "!", '"', "'", "(", ")", ",", "-", ".", ":", ";", "?"] + HEBREW_LETTERS
SPECIAL_TOKENS = ["H", "O", "5"]

ENDINGS_TO_REGULAR = dict(zip("ךםןףץ", "כמנפצ"))


def normalize(c):
    if c in VALID_LETTERS:
        return c
    if c in ENDINGS_TO_REGULAR:
        return ENDINGS_TO_REGULAR[c]
    if c in ["\n", "\t"]:
        return " "
    if c in ["־", "‒", "–", "—", "―", "−"]:
        return "-"
    if c == "[":
        return "("
    if c == "]":
        return ")"
    if c in ["´", "‘", "’"]:
        return "'"
    if c in ["“", "”", "״"]:
        return '"'
    if c.isdigit():
        return "5"
    if c == "…":
        return ","
    if c in ["ײ", "װ", "ױ"]:
        return "H"
    return "O"


class HebrewChar(NamedTuple):
    letter: str
    normalized: str
    dagesh: str
    sin: str
    niqqud: str

    def __str__(self):
        return self.letter + self.dagesh + self.sin + self.niqqud

    def __repr__(self):
        return repr((self.letter, bool(self.dagesh), bool(self.sin), ord(self.niqqud or chr(0))))

    def vocalize(self):
        return self._replace(
            niqqud=vocalize_niqqud(self.niqqud),
            sin=self.sin.replace(RAFE, ""),
            dagesh=vocalize_dagesh(self.letter, self.dagesh),
        )


def items_to_text(items: list[HebrewChar]) -> str:
    return "".join(str(item) for item in items).replace(RAFE, "")


def vocalize_dagesh(letter, dagesh):
    if letter not in "בכפ":
        return ""
    return dagesh.replace(RAFE, "")


def vocalize_niqqud(c):
    # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
    if c in [Niqqud.KAMATZ, Niqqud.PATAKH, Niqqud.REDUCED_PATAKH]:
        return Niqqud.PATAKH

    if c in [Niqqud.HOLAM, Niqqud.REDUCED_KAMATZ]:
        return Niqqud.HOLAM  # TODO: Kamatz-katan

    if c in [Niqqud.SHURUK, Niqqud.KUBUTZ]:
        return Niqqud.KUBUTZ

    if c in [Niqqud.TZEIRE, Niqqud.SEGOL, Niqqud.REDUCED_SEGOL]:
        return Niqqud.SEGOL

    if c == Niqqud.SHVA:
        return ""

    return c.replace(RAFE, "")


def is_hebrew_letter(letter: str) -> bool:
    return "\u05d0" <= letter <= "\u05ea"


def can_dagesh(letter):
    return letter in ("בגדהוזטיכלמנספצקשת" + "ךף")


def can_sin(letter):
    return letter == "ש"


def can_niqqud(letter):
    return letter in ("אבגדהוזחטיכלמנסעפצקרשת" + "ךן")


def can_any(letter):
    return can_niqqud(letter) or can_dagesh(letter) or can_sin(letter)


def iterate_dotted_text(text: str) -> Iterator[HebrewChar]:
    n = len(text)
    text += "  "
    i = 0
    while i < n:
        letter = text[i]

        dagesh = RAFE if can_dagesh(letter) else ""
        sin = RAFE if can_sin(letter) else ""
        niqqud = RAFE if can_niqqud(letter) else ""
        normalized = normalize(letter)
        i += 1

        nbrd = text[i - 15 : i + 15].split()[1:-1]

        assert letter not in ANY_NIQQUD, (
            f"{i}, {nbrd}, {[name_of(c) for word in nbrd for c in word]}"
        )

        if is_hebrew_letter(normalized):
            if text[i] == DAGESH_LETTER:
                # assert dagesh == RAFE, (text[i-5:i+5])
                dagesh = text[i]
                i += 1
            if text[i] in NIQQUD_SIN:
                # assert sin == RAFE, (text[i-5:i+5])
                sin = text[i]
                i += 1
            if text[i] in NIQQUD:
                # assert niqqud == RAFE, (text[i-5:i+5])
                niqqud = text[i]
                i += 1
            if letter == "ו" and dagesh == DAGESH_LETTER and niqqud == RAFE:
                dagesh = RAFE
                niqqud = DAGESH_LETTER

        yield HebrewChar(letter, normalized, dagesh, sin, niqqud)


def split_by_length(characters: Iterable, maxlen: int):
    assert maxlen > 1
    out = []
    space = maxlen
    for c in characters:
        if is_space(c):
            space = len(out)
        out.append(c)
        if len(out) == maxlen - 1:
            yield out[: space + 1]
            out = out[space + 1 :]
    if out:
        yield out


def iterate_file(path):
    with open(path, encoding="utf-8") as f:
        text = "".join(s + " " for s in f.read().split())
        try:
            yield from iterate_dotted_text(text)
        except AssertionError as ex:
            ex.args += (path,)
            raise


def is_space(c):
    if isinstance(c, HebrewChar):
        return c.letter == " "
    elif isinstance(c, str):
        return c == " "
    assert False


class Token:
    def __init__(self, items: list[HebrewChar]):
        self.items = items

    def __str__(self):
        return "".join(str(c) for c in self.items)

    def __repr__(self):
        return "Token(" + repr(self.items) + ")"

    def __lt__(self, other: "Token"):
        return (self.to_undotted(), str(self)) < (other.to_undotted(), str(other))

    def strip_nonhebrew(self) -> "Token":
        start = 0
        end = len(self.items) - 1
        while True:
            if start >= len(self.items):
                return Token([])
            if self.items[start].letter in HEBREW_LETTERS + ANY_NIQQUD:
                break
            start += 1
        while self.items[end].letter not in HEBREW_LETTERS + ANY_NIQQUD:
            end -= 1
        return Token(self.items[start : end + 1])

    def __bool__(self):
        return bool(self.items)

    def __eq__(self, other):
        return self.items == other.items

    @lru_cache
    def to_undotted(self):
        return "".join(str(c.letter) for c in self.items)

    def is_undotted(self):
        return len(self.items) > 1 and all(c.niqqud in [RAFE, ""] for c in self.items)

    def is_definite(self):
        return (
            len(self.items) > 2
            and self.items[0].niqqud == "הַ"[-1]
            and self.items[0].letter in "כבלה"
        )


def tokenize_into(
    tokens_list: list[Token], char_iterator: Iterator[HebrewChar]
) -> Iterator[HebrewChar]:
    current = []
    for c in char_iterator:
        if c.letter.isspace() or c.letter == "-":
            if current:
                tokens_list.append(Token(current).strip_nonhebrew())
            current = []
        else:
            current.append(c)
        yield c
    if current:
        tokens_list.append(Token(current).strip_nonhebrew())


def tokenize(iterator: Iterator[HebrewChar]) -> list[Token]:
    tokens = []
    _ = list(tokenize_into(tokens, iterator))
    return tokens

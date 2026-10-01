"""YallaMorph/CamelMorph aux line builder for the r9 Arabic teacher.

Generates feature-conditional vocalization pairs from paradigm
enumeration (deterministic dictionary resource — not LLM labels):
    input  = "MORPH: <ascii feature tokens> | <undiacritized form>"
    target = "<fully diacritized form>"
"""

from __future__ import annotations

import random
import re
from typing import Iterable, Iterator

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

FORM_MAX_BYTES = 200
FEAT_KEY_ORDER = ("pos", "asp", "per", "gen", "num", "cas", "stt", "voz", "mod")
PERSONS = ("1", "2", "3")
GENDERS = ("m", "f")
NUMBERS = ("s", "d", "p")
CASES = ("nom", "acc", "gen")
STATES = ("d", "i")
VOICES = ("a", "p")
MOODS = ("i", "s", "j", "el", "eh")

PERSON_TOKEN = {"1": "1st", "2": "2nd", "3": "3rd"}


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub("", text)


def feat_tokens(feat) -> str:
    items = feat.items() if hasattr(feat, "items") else feat
    by_key = dict(items)
    ordered = [(k, by_key[k]) for k in FEAT_KEY_ORDER if k in by_key]
    ordered += [(k, v) for k, v in by_key.items() if k not in FEAT_KEY_ORDER]
    return " ".join(f"{k}:{v}" for k, v in ordered)


def aux_pair(feat, form: str) -> tuple[str, str] | None:
    if not form:
        return None
    undiac = strip_diacritics(form)
    if not undiac or len(form.encode("utf-8")) > FORM_MAX_BYTES:
        return None
    return f"MORPH: {feat_tokens(feat)} | {undiac}", form


def iter_verb_specs() -> Iterator[dict]:
    for asp, per, gen, num, voz in (
        (asp, per, gen, num, voz)
        for asp in ("pv", "iv", "c")
        for per in (PERSONS if asp != "c" else ("2",))
        for gen in GENDERS
        for num in NUMBERS
        for voz in VOICES
    ):
        if asp == "iv":
            for mod in MOODS:
                yield {"pos": "v", "asp": asp, "per": PERSON_TOKEN[per], "gen": gen, "num": num, "voz": voz, "mod": mod}
        else:
            yield {"pos": "v", "asp": asp, "per": PERSON_TOKEN[per], "gen": gen, "num": num, "voz": voz}


def _iter_decl_specs(pos: str) -> Iterator[dict]:
    for gen in GENDERS:
        for num in NUMBERS:
            for cas in CASES:
                for stt in STATES:
                    yield {"pos": pos, "gen": gen, "num": num, "cas": cas, "stt": stt}


def iter_noun_specs() -> Iterator[dict]:
    yield from _iter_decl_specs("n")


def iter_adj_specs() -> Iterator[dict]:
    yield from _iter_decl_specs("adj")


def build_lines(rows: Iterable[tuple], cap: int, seed: int = 42) -> list[str]:
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for feat, form in rows:
        p = aux_pair(feat, form)
        if p is None or p in seen:
            continue
        seen.add(p)
        pairs.append(p)
    random.Random(seed).shuffle(pairs)
    return [f"{src}\t{tgt}" for src, tgt in pairs[:cap]]

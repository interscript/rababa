"""Haraqat plane decomposition for the plane-factorized diacritization
model (Stoicheia WO, TODO.sota-2026/04, run-017).

A diacritized string factors into two aligned planes:
    skeleton[i] = base letters (combining marks stripped — the SAME
                  regex as every trainer here, so the skeleton is the
                  exact input the seq2seq rungs see)
    labels[i]   = the exact combining-mark run that follows
                  skeleton[i] in the source text
render() is the byte-exact inverse — canonization happens only when
building the training inventory (plane_classes), never in split, so
round-trip fidelity holds for any convention order.
"""

from __future__ import annotations

import re
from collections import Counter

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

# canonical order: shadda first, then short vowels, tanwin, sukun,
# then anything else (dagger alif, Quranic marks) in encounter order
_ORDER = {"ّ": 0, "َ": 1, "ُ": 2, "ِ": 3, "ً": 4, "ٌ": 5, "ٍ": 6, "ْ": 7}


def canon_combo(marks: str) -> str:
    ranked = sorted(enumerate(marks), key=lambda p: (_ORDER.get(p[1], 8), p[0]))
    return "".join(m for _, m in ranked)


def split_planes(text: str) -> tuple[str, list[str]]:
    """Marks FOLLOW their base letter, so a mark run closes into the
    label of the PREVIOUS skeleton char. Leading marks (before the
    first base char — rare) ride label[0] behind a \\x00 pre-marker."""
    skeleton: list[str] = []
    labels: list[str] = []
    pending: list[str] = []
    leading: list[str] = []
    for ch in text:
        if DIACRITICS_RE.match(ch):
            pending.append(ch)
        else:
            if pending:
                if labels:
                    labels[-1] += "".join(pending)
                else:
                    leading += pending
                pending = []
            skeleton.append(ch)
            labels.append("")
    if pending:
        if labels:
            labels[-1] += "".join(pending)
        else:
            leading += pending
    if leading:
        if labels:
            labels[0] = "".join(leading) + "\x00" + labels[0]
        else:
            # mark-only text (no base letters): nothing to diacritize
            return "", []
    return "".join(skeleton), labels


def render(skeleton: str, labels: list[str]) -> str:
    out: list[str] = []
    for i, base in enumerate(skeleton):
        label = labels[i] if i < len(labels) else ""
        if "\x00" in label:
            pre, post = label.split("\x00", 1)
            out.append(pre)
            out.append(base)
            out.append(post)
        else:
            out.append(base)
            out.append(label)
    return "".join(out)


def plane_classes(texts, min_count: int = 1) -> tuple[list[str], Counter]:
    """Haraqat-combo inventory over a corpus: (classes, counts) with
    classes[0] == '' (none). Combos canonicalized (shadda before the
    vowel); order: none first, then by frequency."""
    counts: Counter = Counter()
    for t in texts:
        _, labels = split_planes(t)
        counts.update(canon_combo(l) for l in labels)
    classes = [""] + [c for c, n in counts.most_common() if c != "" and n >= min_count]
    return classes, counts

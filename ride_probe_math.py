"""Pure math for the RIDE direction-transferability probe.

Input shapes are per-layer mean-pooled hidden states:
    base[layer] = mean-pooled hidden vector of the base checkpoint
    teacher[layer] = same for the improved checkpoint
The probe asks whether the residual direction
    d = teacher - base
computed on one text domain agrees with the direction computed on
another — i.e. whether the SFT delta generalizes or is
domain-idiosyncratic. Kill rule (TODO.sota-2026/05): max-layer cosine
< 0.5 closes the arm.
"""

from __future__ import annotations

import math


def residual_directions(base: list[list[float]], teacher: list[list[float]]) -> list[list[float]]:
    return [[t - b for b, t in zip(brow, trow)] for brow, trow in zip(base, teacher)]


def cosine_rows(a: list[list[float]], b: list[list[float]]) -> list[float]:
    out = []
    for arow, brow in zip(a, b):
        na = math.sqrt(sum(x * x for x in arow))
        nb = math.sqrt(sum(x * x for x in brow))
        if na == 0.0 or nb == 0.0:
            out.append(0.0)
        else:
            out.append(sum(x * y for x, y in zip(arow, brow)) / (na * nb))
    return out

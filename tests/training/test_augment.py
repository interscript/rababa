"""Specs for input augmentation pipeline."""

from __future__ import annotations

import random

from rababa.training.augment import (
    AugmentPipeline,
    CharDropout,
    KeyboardConfusables,
    default_arabic_augment,
    default_hebrew_augment,
)


def test_char_dropout_zero_prob_returns_input_unchanged():
    rng = random.Random(42)
    transform = CharDropout(p=1.0, drop_prob=0.0)
    ids = [1, 2, 3, 4, 5]
    out = transform(ids, rng)
    assert out == ids


def test_char_dropout_high_prob_returns_shorter_sequence():
    rng = random.Random(123)  # different seed — sequence of 10 with high drop
    transform = CharDropout(p=1.0, drop_prob=0.5)
    ids = list(range(1, 21))  # 20 chars, very likely to drop at least one
    out = transform(ids, rng)
    # Most chars dropped — output shorter than input.
    assert len(out) < len(ids)
    # Never returns empty (the impl falls back to original if empty).
    assert len(out) >= 1


def test_keyboard_confusables_swaps_with_specified_alternates():
    rng = random.Random(42)
    transform = KeyboardConfusables(
        p=1.0, swap_prob=1.0,
        confusables={1: [99], 2: [98]},
    )
    ids = [1, 2, 3]
    out = transform(ids, rng)
    # Every char in confusables was swapped.
    assert out == [99, 98, 3]


def test_augment_pipeline_applies_transforms_in_order():
    rng_seed = 42
    pipeline = AugmentPipeline([
        CharDropout(p=1.0, drop_prob=0.0),
        KeyboardConfusables(p=1.0, swap_prob=1.0, confusables={5: [50]}),
    ], seed=rng_seed)
    ids = [5, 6, 7]
    out = pipeline(ids)
    # First transform no-op; second swaps 5→50.
    assert out[0] == 50
    assert out[1] == 6
    assert out[2] == 7


def test_default_arabic_augment_returns_pipeline():
    p = default_arabic_augment()
    assert isinstance(p, AugmentPipeline)
    assert len(p.transforms) == 2  # CharDropout + KeyboardConfusables


def test_default_hebrew_augment_excludes_keyboard_confusables():
    p = default_hebrew_augment()
    assert isinstance(p, AugmentPipeline)
    # Hebrew has no dot-variant confusables — only CharDropout.
    assert len(p.transforms) == 1
    assert isinstance(p.transforms[0], CharDropout)

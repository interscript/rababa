"""Specs for curriculum sampler + phonological features."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from rababa.training.curriculum import (
    CurriculumSampler,
    default_difficulty,
    haraqat_density_difficulty,
)
from rababa.features.arabic import (
    compute_arabic_features,
    features_to_ids,
    FEATURE_VOCAB_SIZES,
)


# ---- Curriculum sampler --------------------------------------------


class _StubDataset(Dataset):
    def __init__(self, n: int = 20) -> None:
        self.examples = [
            type("Ex", (), {"target_ids": list(range(i % 20))})()
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def test_curriculum_sampler_yields_correct_count():
    ds = _StubDataset(n=20)
    sampler = CurriculumSampler(ds, total_epochs=10, current_epoch=0, samples_per_epoch=5)
    indices = list(iter(sampler))
    assert len(indices) == 5


def test_curriculum_sampler_epoch_0_uses_easy_bucket_only():
    ds = _StubDataset(n=20)
    sampler = CurriculumSampler(ds, n_buckets=5, total_epochs=10, current_epoch=0, samples_per_epoch=20)
    indices = list(iter(sampler))
    # At epoch 0, only bucket 0 is accessible.
    bucket_0 = set(sampler.buckets[0])
    for idx in indices:
        assert idx in bucket_0


def test_curriculum_sampler_final_epoch_uses_all_buckets():
    ds = _StubDataset(n=20)
    sampler = CurriculumSampler(ds, n_buckets=5, total_epochs=10, current_epoch=9, samples_per_epoch=20)
    indices = list(iter(sampler))
    all_indices = set().union(*sampler.buckets)
    for idx in indices:
        assert idx in all_indices


def test_curriculum_sampler_set_epoch_updates_progress():
    ds = _StubDataset(n=20)
    sampler = CurriculumSampler(ds, n_buckets=5, total_epochs=10, current_epoch=0)
    assert sampler._max_bucket() == 0
    sampler.set_epoch(5)
    assert sampler._max_bucket() >= 2  # mid-training, more buckets accessible


def test_default_difficulty_returns_finite():
    ex = type("Ex", (), {"target_ids": [1, 2, 3]})()
    d = default_difficulty(ex)
    assert 0.0 <= d <= 1.0


def test_haraqat_density_difficulty_high_for_rare_classes():
    # IDs 9-15 are Shaddah combinations (rare).
    rare_ex = type("Ex", (), {"target_ids": [10, 11, 12, 13]})()
    common_ex = type("Ex", (), {"target_ids": [1, 2, 3, 4]})()
    assert haraqat_density_difficulty(rare_ex) > haraqat_density_difficulty(common_ex)


# ---- Arabic features ----------------------------------------------


def test_compute_arabic_features_word_boundary_at_positions():
    text = "ab cd"
    feats = compute_arabic_features(text)
    assert len(feats) == len(text)
    # Position 0 is word-initial.
    assert feats[0].word_boundary == 1
    # Position 3 (after space) is word-initial.
    assert feats[3].word_boundary == 1
    # Other positions are not.
    assert feats[1].word_boundary == 0


def test_compute_arabic_features_consonant_class_for_sun_letters():
    text = "تث"  # Sun letters
    feats = compute_arabic_features(text)
    assert feats[0].consonant_class == 1  # sun


def test_compute_arabic_features_consonant_class_for_moon_letters():
    text = "اب"  # Moon letters
    feats = compute_arabic_features(text)
    # 'ا' (alif) is not in MOON_LETTERS in our classification — falls to "other"
    # 'ب' is moon
    assert feats[1].consonant_class == 0  # moon


def test_features_to_ids_roundtrip():
    text = "abc"
    feats = compute_arabic_features(text)
    ids = features_to_ids(feats)
    assert set(ids.keys()) == {"iltiqaa", "word_boundary", "consonant_class"}
    for k, v in ids.items():
        assert len(v) == len(text)


def test_feature_vocab_sizes_positive():
    for k, v in FEATURE_VOCAB_SIZES.items():
        assert v >= 2, f"{k} vocab size must be ≥ 2"

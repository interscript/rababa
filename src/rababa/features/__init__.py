"""Phonological features subpackage.

Per-language modules that compute per-character phonological feature IDs
from cleaned input text. Features are passed alongside input_ids and
embedded into the model's first layer.

Open/closed: adding a new language = new file in this subpackage.
"""

from .arabic import (
    FEATURE_VOCAB_SIZES,
    compute_arabic_features,
    features_to_ids,
)

__all__ = ["FEATURE_VOCAB_SIZES", "compute_arabic_features", "features_to_ids"]

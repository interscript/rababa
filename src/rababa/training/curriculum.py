"""Curriculum learning sampler.

Sorts training examples by difficulty (rare-haraqat density for Arabic,
rare-niqqud density for Hebrew, IPA-token-rarity for Thai). The sampler
starts with easy-only and progressively mixes in harder examples.

Why: rare-class examples (e.g. Shaddah+Kasratan in Arabic, ~0.1% of
training) get tiny gradient signal in early epochs when mixed uniformly.
Curriculum learning lets the model first learn the common case solidly,
then refine on rare cases.

Schedule (default `linear`):
  - epoch 0: sample from bucket 0 only (easiest 20%)
  - epoch N/2: sample from buckets 0..N/2
  - epoch N: sample uniformly from all buckets

Open/closed: standalone `Sampler` class. DataLoader accepts any Sampler.
Existing training loops unchanged — just pass `sampler=CurriculumSampler(...)`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from torch.utils.data import Dataset, Sampler


DifficultyFn = Callable[[object], float]


def haraqat_density_difficulty(example) -> float:
    """Difficulty scorer for Arabic examples.

    Returns 0 (easy) to 1 (hard). Based on:
      - Fraction of rare-haraqat positions (Shaddah combinations)
      - Sequence length (longer = harder)
    """
    if not hasattr(example, "target_ids"):
        return 0.0
    targets = example.target_ids
    if not targets:
        return 0.0
    # IDs 9-15 in TARGET_VOCAB are the Shaddah combinations (rare).
    rare_count = sum(1 for t in targets if isinstance(t, int) and 9 <= t <= 15)
    rare_ratio = rare_count / len(targets)
    # Length penalty (normalized).
    length_penalty = min(1.0, len(targets) / 200)
    return min(1.0, 0.7 * rare_ratio + 0.3 * length_penalty)


def default_difficulty(example) -> float:
    """Fallback difficulty scorer — uses target length as proxy."""
    if hasattr(example, "target_ids"):
        return min(1.0, len(example.target_ids) / 100)
    if hasattr(example, "tgt_ids"):
        return min(1.0, len(example.tgt_ids) / 50)
    return 0.5


class CurriculumSampler(Sampler):
    """Sampler that yields examples by difficulty bucket.

    Args:
        dataset: torch Dataset (must support __len__ and __getitem__).
        difficulty_fn: maps example → float in [0, 1].
        n_buckets: number of difficulty buckets (default 5).
        total_epochs: total epochs in training (for schedule).
        current_epoch: which epoch we're in (0-indexed).
        schedule: "linear" (default) or "sqrt".
        samples_per_epoch: how many indices to yield per epoch.
            Defaults to len(dataset).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_fn: DifficultyFn = default_difficulty,
        n_buckets: int = 5,
        total_epochs: int = 20,
        current_epoch: int = 0,
        schedule: str = "linear",
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.n_buckets = n_buckets
        self.total_epochs = max(1, total_epochs)
        self.current_epoch = current_epoch
        self.schedule = schedule
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.rng = random.Random(seed)
        # Pre-compute difficulties and bucket assignment.
        difficulties = [difficulty_fn(dataset[i]) for i in range(len(dataset))]
        # Sort indices by difficulty.
        sorted_pairs = sorted(enumerate(difficulties), key=lambda x: x[1])
        # Bucket boundaries (equal count per bucket).
        bucket_size = max(1, len(sorted_pairs) // n_buckets)
        self.buckets: list[list[int]] = []
        for b in range(n_buckets):
            start = b * bucket_size
            end = (b + 1) * bucket_size if b < n_buckets - 1 else len(sorted_pairs)
            self.buckets.append([idx for idx, _ in sorted_pairs[start:end]])

    def _max_bucket(self) -> int:
        """Highest bucket index accessible this epoch."""
        progress = self.current_epoch / max(1, self.total_epochs - 1)
        if self.schedule == "sqrt":
            progress = math.sqrt(progress)
        # Linear: epoch 0 → bucket 0 only, last epoch → all buckets.
        return min(self.n_buckets - 1, int(progress * self.n_buckets))

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch — call at the start of each epoch."""
        self.current_epoch = max(0, min(self.total_epochs - 1, epoch))

    def __iter__(self):
        max_b = self._max_bucket()
        # Sample uniformly from buckets 0..max_b.
        accessible: list[int] = []
        for b in range(max_b + 1):
            accessible.extend(self.buckets[b])
        # Yield samples_per_epoch indices, sampled with replacement if needed.
        for _ in range(self.samples_per_epoch):
            yield self.rng.choice(accessible)

    def __len__(self) -> int:
        return self.samples_per_epoch

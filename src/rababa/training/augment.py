"""Input-side augmentation policies.

Composable transforms applied at DataLoader-time (not pre-baked into
the dataset). Each transform takes a list[int] of input IDs and returns
a (possibly modified) list[int] of the same length.

`AugmentPipeline` chains transforms and applies them with given
probabilities. The pipeline is data-dependent (passed per dataset),
not baked into the model — OCP.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from ..constants import MASK_ID, PAD_ID


class AugmentTransform(Callable):
    """Base class for input-side augmentation transforms."""

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {p}")
        self.p = p

    def __call__(self, input_ids: list[int], rng: random.Random) -> list[int]:
        if rng.random() < self.p:
            return self.apply(input_ids, rng)
        return list(input_ids)

    def apply(self, input_ids: list[int], rng: random.Random) -> list[int]:
        raise NotImplementedError


class CharDropout(AugmentTransform):
    """Drop characters at random positions (replace with PAD)."""

    def __init__(self, p: float = 0.5, drop_prob: float = 0.05) -> None:
        super().__init__(p)
        self.drop_prob = drop_prob

    def apply(self, input_ids: list[int], rng: random.Random) -> list[int]:
        out: list[int] = []
        for cid in input_ids:
            if cid != PAD_ID and rng.random() < self.drop_prob:
                # Skip the char entirely (sequence gets shorter).
                continue
            out.append(cid)
        # Ensure we don't return an empty sequence.
        return out if out else list(input_ids)


class KeyboardConfusables(AugmentTransform):
    """Replace chars with keyboard-adjacent confusables.

    For Arabic: pairs like (ب, ت, ث) — same base shape, different dot count.
    For Hebrew: pairs like (ב, כ) — visually similar.

    The confusable map is set on the transform instance. Pass a per-language
    dict at construction time.
    """

    def __init__(
        self,
        p: float = 0.5,
        swap_prob: float = 0.02,
        confusables: dict[int, list[int]] | None = None,
    ) -> None:
        super().__init__(p)
        self.swap_prob = swap_prob
        self.confusables = confusables or _DEFAULT_ARABIC_CONFUSABLES

    def apply(self, input_ids: list[int], rng: random.Random) -> list[int]:
        out: list[int] = []
        for cid in input_ids:
            if cid in self.confusables and rng.random() < self.swap_prob:
                out.append(rng.choice(self.confusables[cid]))
            else:
                out.append(cid)
        return out


# Default int-keyed confusables map. Empty by default — callers populate
# from the encoder's vocab. Using string keys would require the encoder
# at module-import time, which couples this module to the language.
_DEFAULT_ARABIC_CONFUSABLES: dict[int, list[int]] = {}


class AugmentPipeline:
    """Compose multiple AugmentTransforms.

    Each transform is applied in order; the output of one is the input
    to the next. A shared RNG ensures reproducibility per epoch.
    """

    def __init__(self, transforms: Iterable[AugmentTransform], seed: int = 42) -> None:
        self.transforms = list(transforms)
        self.rng = random.Random(seed)

    def __call__(self, input_ids: list[int]) -> list[int]:
        x = list(input_ids)
        for t in self.transforms:
            x = t(x, self.rng)
        return x


def default_arabic_augment(seed: int = 42) -> AugmentPipeline:
    """Standard Arabic augmentation: light char dropout + keyboard confusables."""
    return AugmentPipeline([
        CharDropout(p=1.0, drop_prob=0.05),
        KeyboardConfusables(p=1.0, swap_prob=0.02),
    ], seed=seed)


def default_hebrew_augment(seed: int = 42) -> AugmentPipeline:
    """Standard Hebrew augmentation: light char dropout only.

    Hebrew keyboard confusables are rare (no dot-count variants), so
    we skip that transform.
    """
    return AugmentPipeline([
        CharDropout(p=1.0, drop_prob=0.05),
    ], seed=seed)

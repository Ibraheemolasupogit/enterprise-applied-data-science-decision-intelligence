"""Deterministic seeding shared by every pillar's reproducibility story.

Centralising seeding here means each pillar's "this is reproducible"
claim rests on one audited code path instead of N ad hoc reseeds.
"""

from __future__ import annotations

import random

import numpy as np


def seeded_random(seed: int) -> random.Random:
    """Return a ``random.Random`` instance deterministically seeded.

    Args:
        seed: Integer seed. Same seed always yields the same sequence.
    """
    return random.Random(seed)


def seeded_generator(seed: int) -> np.random.Generator:
    """Return a NumPy ``Generator`` deterministically seeded.

    Args:
        seed: Integer seed. Same seed always yields the same sequence.
    """
    return np.random.default_rng(seed)


def spawn_generators(seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent, deterministic NumPy generators derived from ``seed``.

    Multi-stage synthetic-data pipelines use one spawned generator per stage
    instead of reusing a single generator, so that adding, removing or
    reordering an unrelated stage never changes another stage's draws.
    """
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)]

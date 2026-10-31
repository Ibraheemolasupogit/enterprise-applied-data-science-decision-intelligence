"""Deterministic seeding shared by every pillar's reproducibility story.

Centralising seeding here means each pillar's "this is reproducible"
claim rests on one audited code path instead of N ad hoc reseeds.
"""

from __future__ import annotations

import random


def seeded_random(seed: int) -> random.Random:
    """Return a ``random.Random`` instance deterministically seeded.

    Args:
        seed: Integer seed. Same seed always yields the same sequence.
    """
    return random.Random(seed)

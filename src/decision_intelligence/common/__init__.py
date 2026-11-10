"""Shared utilities used across pillar modules.

This package is the single home for cross-pillar primitives: seeding (M1),
input validation and the bootstrap confidence-interval helper (M3).
Point-in-time utilities live in ``decision_intelligence.data.point_in_time``
(M2). Pillar modules must depend on these owners rather than reimplementing
shared logic locally — see docs/architecture/overview.md.
"""

from decision_intelligence.common.bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_two_sample_ci,
)
from decision_intelligence.common.seeding import seeded_generator, seeded_random, spawn_generators

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "bootstrap_two_sample_ci",
    "seeded_generator",
    "seeded_random",
    "spawn_generators",
]

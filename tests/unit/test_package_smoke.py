"""Milestone 1 smoke tests: package installs and seeding is deterministic.

Substantial statistical/method tests belong to the milestones that
introduce those methods (M3+). This file only proves the foundation
is reproducible end to end.
"""

from decision_intelligence import __version__
from decision_intelligence.common import seeded_random


def test_package_exposes_a_version() -> None:
    assert __version__ == "0.1.0"


def test_seeded_random_is_deterministic() -> None:
    first = seeded_random(42).random()
    second = seeded_random(42).random()
    assert first == second


def test_seeded_random_differs_across_seeds() -> None:
    assert seeded_random(1).random() != seeded_random(2).random()

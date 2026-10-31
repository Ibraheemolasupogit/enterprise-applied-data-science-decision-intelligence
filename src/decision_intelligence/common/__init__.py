"""Shared utilities used across pillar modules.

This package is the single home for cross-pillar primitives (seeding,
and later: bootstrap/CI helpers, point-in-time utilities). Pillar
modules must depend on this package rather than reimplementing shared
logic locally — see docs/architecture/overview.md.
"""

from decision_intelligence.common.seeding import seeded_random

__all__ = ["seeded_random"]

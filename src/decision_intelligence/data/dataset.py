"""The bundle returned by one synthetic data build."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.ground_truth import GroundTruth


@dataclass(frozen=True)
class SyntheticDataset:
    """Everything produced by one deterministic synthetic data build.

    ``tables`` holds every table a real NexaMarket pipeline would actually
    have. ``latent_affinities`` and ``ground_truth`` are evaluation-only
    knowledge that a real system would never have access to — later
    pillars may read them to check their own estimates, but must never
    use them as model inputs.
    """

    config: SyntheticDataConfig
    tables: dict[str, pd.DataFrame]
    latent_affinities: pd.DataFrame
    ground_truth: GroundTruth

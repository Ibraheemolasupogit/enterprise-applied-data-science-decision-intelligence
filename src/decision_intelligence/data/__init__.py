"""Synthetic data foundation: contracts, generators and the build pipeline.

Every pillar module consumes data through this package rather than
generating its own — see docs/methodology/synthetic-data.md.
"""

from decision_intelligence.data.config import DataQualityConfig, SyntheticDataConfig
from decision_intelligence.data.dataset import SyntheticDataset
from decision_intelligence.data.pipeline import build_synthetic_dataset
from decision_intelligence.data.validation import validate_dataset

__all__ = [
    "DataQualityConfig",
    "SyntheticDataConfig",
    "SyntheticDataset",
    "build_synthetic_dataset",
    "validate_dataset",
]

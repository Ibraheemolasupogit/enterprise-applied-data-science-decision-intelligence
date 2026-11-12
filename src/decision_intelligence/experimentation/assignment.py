"""Random assignment: deterministic hashing, complete and stratified randomisation.

Two families with different guarantees:

* **Hash-based Bernoulli assignment** (:func:`hash_assignment`) — each unit's
  arm is a pure function of ``(experiment, salt, unit_id)`` via SHA-256, so
  it is reproducible across processes, independent of the order or size of
  the population, stable as new units arrive, and statistically independent
  between experiments with different names/salts. Arm sizes are random
  (multinomial), which is exactly the null distribution the SRM test uses.
* **Complete / stratified randomisation** (:func:`complete_randomisation`,
  :func:`stratified_randomisation`) — a seeded permutation fixes the arm
  sizes (overall or within each stratum). Every unit still has marginal
  probability ``w_a`` of arm ``a``: remainders are allocated by systematic
  sampling with inclusion probabilities equal to the fractional parts, so
  no arm is favoured by rounding. Stratifying on prognostic pre-treatment
  variables removes chance imbalance in them; it must then be reflected in
  the analysis (:func:`..analysis.stratified_difference`) or the reported
  uncertainty is conservative.

Cluster randomisation (:func:`cluster_randomisation`) assigns whole
clusters; the analysis must then be clustered.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from decision_intelligence.common.validation import FloatArray

_TWO_64 = float(2**64)


def _validate_weights(arms: Sequence[str], weights: Sequence[float] | None) -> FloatArray:
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("need at least two distinct arms")
    w = np.full(len(arms), 1.0 / len(arms)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(arms),) or (w <= 0).any() or abs(float(w.sum()) - 1.0) > 1e-9:
        raise ValueError("weights must be positive, one per arm, and sum to 1")
    return w


def hash_unit(unit_id: object, *, experiment: str, salt: str = "") -> float:
    """Uniform number in [0, 1) from SHA-256 of ``"{experiment}|{salt}|{unit_id}"``."""
    digest = hashlib.sha256(f"{experiment}|{salt}|{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _TWO_64


def hash_assignment(
    unit_ids: ArrayLike,
    *,
    experiment: str,
    arms: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] | None = None,
    salt: str = "",
) -> np.ndarray:
    """Arm label per unit by hash bucketing (Bernoulli / multinomial assignment)."""
    if not experiment:
        raise ValueError("experiment name is required (it decorrelates experiments)")
    w = _validate_weights(arms, weights)
    edges = np.cumsum(w)
    edges[-1] = 1.0
    ids = np.asarray(unit_ids)
    u = np.array([hash_unit(i, experiment=experiment, salt=salt) for i in ids.tolist()])
    index = np.searchsorted(edges, u, side="right")
    return np.asarray(arms, dtype=object)[np.minimum(index, len(arms) - 1)]


def _arm_counts(n: int, w: FloatArray, rng: np.random.Generator) -> np.ndarray:
    """Integer counts with E[count_a] = n w_a exactly (systematic sampling of remainders)."""
    target = n * w
    counts = np.floor(target).astype(int)
    fractions = target - counts
    remainder = int(round(float(fractions.sum())))
    if remainder:
        cumulative = np.concatenate([[0.0], np.cumsum(fractions)])
        points = rng.random() + np.arange(remainder)
        chosen = np.minimum(np.searchsorted(cumulative, points, side="right") - 1, w.size - 1)
        counts[chosen] += 1
    return counts


def complete_randomisation(
    n: int,
    *,
    rng: np.random.Generator,
    arms: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Seeded random permutation with (expected-)exact arm sizes."""
    if n < len(arms):
        raise ValueError("need at least one unit per arm")
    w = _validate_weights(arms, weights)
    labels = np.repeat(np.asarray(arms, dtype=object), _arm_counts(n, w, rng))
    return np.asarray(rng.permutation(labels))


def stratified_randomisation(
    strata: ArrayLike,
    *,
    rng: np.random.Generator,
    arms: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Complete randomisation within each stratum (blocks processed in sorted order)."""
    labels_by_unit = np.asarray(strata)
    w = _validate_weights(arms, weights)
    out = np.empty(labels_by_unit.size, dtype=object)
    for stratum in sorted(set(labels_by_unit.tolist()), key=str):
        index = np.flatnonzero(labels_by_unit == stratum)
        labels = np.repeat(np.asarray(arms, dtype=object), _arm_counts(index.size, w, rng))
        out[index] = rng.permutation(labels)
    return out


def cluster_randomisation(
    clusters: ArrayLike,
    *,
    rng: np.random.Generator,
    arms: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Assign whole clusters by complete randomisation; returns one label per unit."""
    ids = np.asarray(clusters)
    unique = np.unique(ids)
    if unique.size < 2 * len(arms):
        raise ValueError("cluster randomisation needs at least two clusters per arm")
    cluster_labels = complete_randomisation(unique.size, rng=rng, arms=arms, weights=weights)
    return np.asarray(cluster_labels[np.searchsorted(unique, ids)])


__all__ = [
    "cluster_randomisation",
    "complete_randomisation",
    "hash_assignment",
    "hash_unit",
    "stratified_randomisation",
]

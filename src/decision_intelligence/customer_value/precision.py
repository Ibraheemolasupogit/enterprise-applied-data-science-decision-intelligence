"""Precision planning for the Milestone 11 comparisons (P0).

**Paired squared-error differences.** For predictions ``m1, m2`` made from the information at
the cutoff, outcome ``R = mu + eps`` with ``mu = E[R | F_C]`` and ``E[eps | F_C] = 0``, the
per-customer loss difference is

    D_i = (R - m1)^2 - (R - m2)^2 = delta (2 eps + 2 mu - m1 - m2),   delta = m2 - m1.

When ``m1 = mu`` (a near-oracle model) and ``m2 = mu + delta`` with ``delta`` independent of
``eps`` given ``F_C``: ``E[D] = E[delta^2] = tau^2`` and
``Var(D) = 4 tau^2 sigma^2 + Var(delta^2)``, where ``sigma^2 = E[Var(R | F_C)]``. Ignoring the
second term (small when ``tau << sigma``), ``SE(mean D) = 2 tau sigma / sqrt(n_eff)``. A
difference is detectable at two-sided level ``alpha`` with power ``1 - beta`` when
``E[D] >= k SE``, ``k = z_{1-alpha/2} + z_{1-beta}``; with ``E[D] = tau^2`` this gives the
**minimum detectable relative excess MSE** ``tau^2 / sigma^2 >= 4 k^2 / n_eff``.

**Customer clustering (Kish design effect).** Pooling cutoffs puts the same customer in
several rows. For the row mean of ``n`` rows in clusters of sizes ``m_g`` (``sum m_g = n``),
with exchangeable within-customer correlation ``rho``,
``Var(mean) = sigma^2 (n + rho (sum m_g^2 - n)) / n^2``, so the design effect is
``DE = 1 + rho (sum m_g^2 / n - 1)`` and ``n_eff = n / DE``. At ``rho = 1``,
``n_eff = n^2 / sum m_g^2`` — which equals the number of customers only when every customer
has the same number of rows; with unequal sizes it is smaller.

*Clustering never lowers the minimum detectable difference.* Relative to treating every row
as independent (``rho = 0``, ``n_eff = n``), clustering raises it. Pooling two cutoffs can
still give a smaller detectable difference than one cutoff alone, because the pooled set has
more distinct customers and their second rows add partly independent information; that is
an effect of pooling, not of clustering.

**Coverage studies are reported, not enforced.** With ``r`` replications and nominal level
``p`` the Monte Carlo standard error is ``sqrt(p (1 - p) / r)``. :func:`classify_coverage`
labels an observed coverage as consistent with, below or above nominal by the exact two-sided
binomial test; the label is a reporting criterion. No test asserts a coverage level, and a
coverage result never changes the protocol after main-seed results are seen.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def critical_sum(alpha: float, power: float) -> float:
    """``z_{1-alpha/2} + z_{power}``."""
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie in (0, 1)")
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def paired_mse_difference_se(tau: float, sigma: float, n_eff: float) -> float:
    """Leading-order SE of a mean paired squared-error difference (see module docstring)."""
    if tau < 0 or sigma < 0 or n_eff <= 0:
        raise ValueError("tau and sigma must be non-negative and n_eff positive")
    return 2.0 * tau * sigma / math.sqrt(n_eff)


def minimum_detectable_relative_excess(n_eff: float, alpha: float, power: float) -> float:
    """Smallest ``tau^2 / sigma^2`` detectable against a near-oracle model: ``4 k^2 / n_eff``."""
    if n_eff <= 0:
        raise ValueError("n_eff must be positive")
    k = critical_sum(alpha, power)
    return 4.0 * k * k / n_eff


def required_n_eff(relative_excess: float, alpha: float, power: float) -> float:
    """Effective sample size needed to detect a relative excess MSE ``tau^2 / sigma^2``."""
    if relative_excess <= 0:
        raise ValueError("relative_excess must be positive")
    k = critical_sum(alpha, power)
    return 4.0 * k * k / relative_excess


def _sizes(cluster_sizes: ArrayLike) -> np.ndarray:
    m = np.asarray(cluster_sizes, dtype=np.float64)
    if m.ndim != 1 or m.size == 0 or np.any(m < 1) or np.any(m != np.floor(m)):
        raise ValueError("cluster sizes must be a non-empty vector of positive integers")
    return m


def design_effect(cluster_sizes: ArrayLike, rho: float) -> float:
    """Kish design effect ``1 + rho (sum m_g^2 / n - 1)`` for exchangeable correlation."""
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    m = _sizes(cluster_sizes)
    n = float(m.sum())
    return 1.0 + rho * (float(np.sum(m * m)) / n - 1.0)


def effective_n(cluster_sizes: ArrayLike, rho: float) -> float:
    """``n / design_effect``; ``n^2 / sum m_g^2`` at ``rho = 1``."""
    m = _sizes(cluster_sizes)
    return float(m.sum()) / design_effect(m, rho)


def coverage_mc_se(level: float, replications: int) -> float:
    if not 0 < level < 1 or replications < 1:
        raise ValueError("level in (0, 1) and replications >= 1 required")
    return math.sqrt(level * (1 - level) / replications)


def coverage_acceptance_region(level: float, replications: int, alpha: float = 0.05
                               ) -> tuple[float, float]:
    """Smallest and largest observed coverage proportions not rejected by the exact two-sided
    binomial test of ``H0: coverage = level`` at ``alpha``."""
    if not 0 < level < 1 or replications < 1 or not 0 < alpha < 1:
        raise ValueError("invalid level, replications or alpha")
    accepted = [k for k in range(replications + 1)
                if stats.binomtest(k, replications, level).pvalue >= alpha]
    return accepted[0] / replications, accepted[-1] / replications


def classify_coverage(covered: int, replications: int, level: float, alpha: float = 0.05
                      ) -> str:
    """Reporting label: ``consistent with nominal``, ``below nominal`` or ``above nominal``."""
    if not 0 <= covered <= replications:
        raise ValueError("covered must lie in [0, replications]")
    if stats.binomtest(covered, replications, level).pvalue >= alpha:
        return "consistent with nominal"
    return "below nominal" if covered / replications < level else "above nominal"

"""M11 P4: E3 recovers the parameters of data generated from its own model (correct
specification): landmark rows, person-months, merged tenure bands and the standardised
penalised design all on the estimation path (development seeds only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.customer_value import survival as sv

W = 12


def _simulate(seed: int, n: int) -> tuple[pd.DataFrame, np.ndarray, dict[int, float]]:
    """Landmark rows with Tier A registry columns; ``logit h = gamma(band of age) + 0.6
    [segment = premium] + 0.25 mean_contribution`` (raw scale); every other column null."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(2, 30, n)
    feats = pd.DataFrame({
        "segment": rng.choice(["standard", "premium", "value"], n, p=[0.5, 0.25, 0.25]),
        "channel": rng.choice(["organic", "paid_search"], n),
        "tenure_months": tenure, "acquisition_cohort": rng.integers(0, 20, n),
        "promo_months_remaining": 0, "paid_months": tenure + 1,
        "mean_contribution": rng.normal(7.0, 1.5, n),
        "last_contribution": rng.normal(7.0, 1.5, n)})
    gamma = {0: -3.2, 1: -3.2, 2: -3.2, 3: -3.4, 4: -3.6, 5: -3.9, 6: -4.1}  # by band
    base = 0.6 * (feats["segment"] == "premium").to_numpy() + 0.25 * (
        feats["mean_contribution"].to_numpy() - 7.0)
    n_paid = np.full(n, W)
    alive = np.ones(n, dtype=bool)
    for k in range(1, W + 1):
        band = sv.tenure_band(tenure + k - 1)
        eta = np.array([gamma[int(b)] for b in band]) + base
        churn = alive & (rng.random(n) < 1 / (1 + np.exp(-eta)))
        n_paid[churn] = k - 1
        alive &= ~churn
    return feats, n_paid, gamma


def test_e3_recovers_its_own_generating_parameters() -> None:
    feats, n_paid, gamma = _simulate(4941, 60_000)
    fit = sv.fit_hazard(feats, n_paid, W, "A", penalty=0.01)
    assert fit.status == "ok"
    assert fit.bands == ("[0, 3)", "[3, 6)", "[6, 12)", "[12, 24)", "[24, inf)")
    d = fit.design
    scale = dict(zip(d.standardiser.names, d.standardiser.scale, strict=True))
    mean = dict(zip(d.standardiser.names, d.standardiser.mean, strict=True))
    true_beta = {c: 0.0 for c in d.columns}
    true_beta["mean_contribution"] = 0.25 * scale["mean_contribution"]
    # segment indicators against the most frequent level ("standard")
    assert d.levels["segment"][1] == "standard"
    true_beta["segment=premium"] = 0.6 * scale["segment=premium"]
    # the band intercepts absorb the centring of every penalised column
    offset = sum(true_beta[c] * mean[c] / scale[c] for c in d.columns) - 0.25 * 7.0
    # merged groups [0,3) [3,6) [6,12) [12,24) [24,inf) = bands 2..6 (ages 0-1 never occur)
    true = np.r_[np.array([gamma[b] for b in (2, 3, 4, 5, 6)]) + offset,
                 [true_beta[c] for c in d.columns]]
    est = fit.result.coefficients
    se = np.sqrt(np.diag(fit.result.covariance))
    z = (est - true) / se
    assert np.all(np.abs(z) <= 3.5), dict(zip(fit.bands + d.columns, np.round(z, 2),
                                              strict=True))
    assert float(np.sum(z * z)) <= stats.chi2.ppf(0.999, z.size)

"""M11 P6: E5 recovers the parameters of data generated from its own two-part model, and the S2
pipeline (E5 fit, deterministic customer split, conformal quantile) attains its exact
finite-sample coverage on exchangeable customers (development seeds only)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from decision_intelligence.customer_value import predictive as pr
from decision_intelligence.statistics.conformal import conformal_quantile, conformal_rank
from decision_intelligence.statistics.gamma_regression import GammaRegressionResult


def _frame(rng: np.random.Generator, n: int, start_id: int = 1
           ) -> tuple[pd.DataFrame, np.ndarray]:
    """Tier A contractual registry columns; logit P(R > 0) = 1.5 + 0.8 [premium] + 0.4 z;
    R | R > 0 ~ Gamma(shape 3, mean exp(3.8 + 0.25 z)), z = mean_contribution - 7."""
    tenure = rng.integers(2, 30, n)
    seg = rng.choice(["standard", "premium", "value"], n, p=[0.5, 0.25, 0.25])
    mc = rng.normal(7.0, 1.5, n)
    frame = pd.DataFrame({
        "segment": seg, "channel": rng.choice(["organic", "paid_search"], n),
        "tenure_months": tenure, "acquisition_cohort": rng.integers(0, 20, n),
        "promo_months_remaining": 0, "paid_months": tenure + 1, "mean_contribution": mc,
        "last_contribution": rng.normal(7.0, 1.5, n)},
        index=pd.Index(np.arange(start_id, start_id + n), name="customer_id"))
    z = mc - 7.0
    pos = rng.random(n) < 1 / (1 + np.exp(-(1.5 + 0.8 * (seg == "premium") + 0.4 * z)))
    mean = np.exp(3.8 + 0.25 * z)
    r = np.where(pos, rng.gamma(3.0, mean / 3.0), 0.0)
    return frame.assign(realised=r), r


def test_e5_recovers_its_own_generating_parameters() -> None:
    rng = np.random.default_rng(4991)
    frame, r = _frame(rng, 40_000)
    p1, p2 = pr.fit_parts(frame, "contractual", "A", penalty=0.01)
    assert p1.status == "ok" and p2.status == "ok"
    for part, truth in ((p1, {"mean_contribution": 0.4, "segment=premium": 0.8}),
                        (p2, {"mean_contribution": 0.25})):
        d = part.design
        scale = dict(zip(d.standardiser.names, d.standardiser.scale, strict=True))
        true = np.array([truth.get(c, 0.0) * scale[c] for c in d.columns])
        est = part.result.coefficients[1:]
        se = np.sqrt(np.diag(part.result.covariance))[1:]
        z = (est - true) / se
        assert np.all(np.abs(z) <= 3.5), dict(zip(d.columns, np.round(z, 2), strict=True))
    assert isinstance(p2.result, GammaRegressionResult)
    assert math.isfinite(p2.result.shape)
    assert abs(math.log(p2.result.shape) - math.log(3.0)) < 0.05


def test_s2_pipeline_has_exact_finite_sample_coverage_on_exchangeable_customers() -> None:
    """One landmark per customer, i.i.d. customers: the calibration customers (chosen by S2's
    hash split) and new customers are exchangeable, so P(cover) = E[ceil((n+1) level)/(n+1)]
    over the random calibration size n; checked within 4 SE across replications."""
    rng = np.random.default_rng(4992)
    reps, n_cust, n_new = 300, 400, 200
    cover: dict[float, list[float]] = {0.8: [], 0.95: []}
    expected: dict[float, list[float]] = {0.8: [], 0.95: []}
    for rep in range(reps):
        frame, r = _frame(rng, n_cust, start_id=1 + rep * 10_000)
        cal = pr.calibration_customers(frame.index.to_numpy())
        p1, p2 = pr.fit_parts(frame.loc[~cal], "contractual", "A", penalty=1.0)
        p, m = pr._predict_parts(p1, p2, frame.loc[cal])
        scores = np.abs(r[cal] - p * m)
        new, r_new = _frame(rng, n_new, start_id=5_000_000 + rep * 10_000)
        pn, mn = pr._predict_parts(p1, p2, new)
        for level in cover:
            q = conformal_quantile(scores, level)
            cover[level].append(float(np.mean(np.abs(r_new - pn * mn) <= q)))
            expected[level].append(conformal_rank(int(cal.sum()), level)
                                   / (int(cal.sum()) + 1))
    for level in cover:
        c = np.asarray(cover[level])
        target = float(np.mean(expected[level]))
        se = float(c.std(ddof=1)) / math.sqrt(reps)
        assert abs(float(c.mean()) - target) <= 4 * se, (level, c.mean(), target, se)
        assert target >= level

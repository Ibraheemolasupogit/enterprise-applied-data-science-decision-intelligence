"""M11 P7: predictive metrics against brute-force references (development seeds only)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from decision_intelligence.statistics import predictive_metrics as pm


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5801)
    pred = rng.gamma(2.0, 10.0, 400)
    y = pred * rng.gamma(1.5, 1 / 1.5, 400) + rng.normal(0, 2, 400)
    pred[:20] = pred[20]  # ties
    return pred, y


def test_point_metrics_match_brute_force(data: tuple[np.ndarray, np.ndarray]) -> None:
    p, y = data
    ref_bias = float(np.sum(p)) / float(np.sum(y)) - 1
    assert pm.relative_bias(p, y) == pytest.approx(ref_bias, rel=1e-12)
    assert pm.mae(p, y) == pytest.approx(np.mean([abs(a - b) for a, b in zip(y, p, strict=True)]))
    assert pm.rmse(p, y) == pytest.approx(math.sqrt(np.mean([(a - b) ** 2 for a, b in
                                                              zip(y, p, strict=True)])))
    line = pm.calibration_line(p, y)
    ref = sm.OLS(y, sm.add_constant(p)).fit(cov_type="HC1")
    assert line["intercept"] == pytest.approx(ref.params[0], rel=1e-10)
    assert line["slope"] == pytest.approx(ref.params[1], rel=1e-10)
    assert line["slope_se"] == pytest.approx(ref.bse[1], rel=1e-8)
    assert line["intercept_se"] == pytest.approx(ref.bse[0], rel=1e-8)
    assert pm.spearman(p, y) == pytest.approx(stats.spearmanr(p, y).statistic, rel=1e-12)


def test_ranking_metrics_match_brute_force(data: tuple[np.ndarray, np.ndarray]) -> None:
    p, y = data
    n_top = math.ceil(0.1 * p.size)
    order = sorted(range(p.size), key=lambda i: (-p[i], i))
    assert pm.top_decile_capture(p, y) == pytest.approx(sum(y[i] for i in order[:n_top])
                                                        / y.sum())
    groups = pm.decile_groups(p)
    ranks = sorted(range(p.size), key=lambda i: (p[i], i))
    for pos, i in enumerate(ranks):
        assert groups[i] == pos * 10 // p.size
    rel = pm.decile_reliability(p, y)
    for g in range(10):
        sel = groups == g
        assert rel.loc[g, "mean_pred"] == pytest.approx(p[sel].mean())
        assert rel.loc[g, "mean_y"] == pytest.approx(y[sel].mean())

    def lorenz_gini(key: np.ndarray) -> float:  # area between the cumulative-share curve and
        idx = sorted(range(p.size), key=lambda i: (-key[i], i))  # the diagonal, by loops
        total, acc, area = y.sum(), 0.0, 0.0
        for k, i in enumerate(idx, start=1):
            acc += y[i]
            area += acc / total - k / p.size
        return area / p.size

    assert pm.normalised_gini(p, y) == pytest.approx(lorenz_gini(p) / lorenz_gini(y))
    assert pm.normalised_gini(y, y) == pytest.approx(1.0)


def _survival_sample(seed: int, n: int, censor: bool) -> tuple[np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = 12
    haz = rng.uniform(0.02, 0.2, n)
    T = rng.geometric(haz) - 1  # months survived
    C = rng.integers(0, h + 1, n) if censor else np.full(n, h)
    C = np.minimum(C, h)
    event = T < C
    time = np.where(event, T, C)
    surv = (1 - haz[:, None]) ** np.arange(1, h + 1)[None, :]  # S(t) = P(T >= t)
    return time, event, surv, haz


def test_censoring_km_and_ipcw_brier_match_brute_force() -> None:
    time, event, surv, _ = _survival_sample(5802, 300, True)
    g = pm.censoring_km(time, event, 12)
    for u in range(13):
        ref = 1.0
        for s in range(u + 1):
            at_risk = sum(1 for t in time if t >= s)
            cens = sum(1 for t, e in zip(time, event, strict=True) if t == s and not e)
            if at_risk:
                ref *= 1 - cens / at_risk
        assert g[u] == pytest.approx(ref)
    for t_eval in (1, 6, 12):
        total = 0.0
        for i in range(time.size):
            s = surv[i, t_eval - 1]
            if time[i] < t_eval and event[i]:
                total += s ** 2 / g[time[i]]
            elif time[i] >= t_eval:
                total += (1 - s) ** 2 / g[t_eval - 1]
        assert pm.ipcw_brier(t_eval, time, event, surv[:, t_eval - 1]) == pytest.approx(
            total / time.size)
    ib = np.mean([pm.ipcw_brier(k, time, event, surv[:, k - 1]) for k in range(1, 13)])
    assert pm.integrated_brier(time, event, surv) == pytest.approx(ib)


def test_concordance_matches_brute_force() -> None:
    time, event, surv, haz = _survival_sample(5803, 250, True)
    risk = np.round(haz, 2)  # ties in risk
    g = pm.censoring_km(time, event, 12)
    num = den = unum = uden = 0.0
    for i in range(time.size):
        if not event[i]:
            continue
        for j in range(time.size):
            if time[j] > time[i]:
                c = 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
                num += c
                den += 1
                if time[i] < 10:
                    w = 1 / g[time[i]] ** 2
                    unum += w * c
                    uden += w
    assert pm.harrell_c(time, event, risk) == pytest.approx(num / den)
    assert pm.uno_c(time, event, risk, tau=10) == pytest.approx(unum / uden)


def test_d_calibration_exact_mass_and_correct_model() -> None:
    time, event, surv, _ = _survival_sample(5804, 4000, False)
    out = pm.d_calibration(time, event, surv)
    assert out["p_value"] > 1e-3  # the true model is D-calibrated (exact mass, no noise)
    assert 0.08 < out["min_bin_share"] <= out["max_bin_share"] < 0.12
    wrong = pm.d_calibration(time, event, surv ** 3)
    assert wrong["p_value"] < 1e-6


def test_count_scores_match_brute_force() -> None:
    rng = np.random.default_rng(5805)
    pmf = rng.dirichlet(np.ones(6), 50)
    y = rng.integers(0, 6, 50)
    ls = pm.log_score(pmf, y)
    crps = pm.crps_counts(pmf, y)
    for i in range(50):
        assert ls[i] == pytest.approx(-math.log(pmf[i, y[i]]))
        cdf = np.cumsum(pmf[i])
        assert crps[i] == pytest.approx(sum((cdf[k] - (y[i] <= k)) ** 2 for k in range(6)))
        # CRPS = E|X - y| - E|X - X'| / 2 for a discrete X
        e1 = sum(pmf[i, k] * abs(k - y[i]) for k in range(6))
        e2 = sum(pmf[i, a] * pmf[i, b] * abs(a - b) for a in range(6) for b in range(6))
        assert crps[i] == pytest.approx(e1 - e2 / 2)
    pit = pm.randomised_pit(pmf, y, np.random.default_rng(5806))
    cdf = np.cumsum(pmf, axis=1)
    lo = np.where(y > 0, cdf[np.arange(50), np.maximum(y - 1, 0)], 0.0)
    assert np.all((pit >= lo - 1e-12) & (pit <= cdf[np.arange(50), y] + 1e-12))
    paid = np.array([[0.9, 0.8, 0.5], [1.0, 1.0, 1.0]])
    assert pm.paid_months_pmf(paid) == pytest.approx(np.array([[0.1, 0.1, 0.3, 0.5],
                                                                [0.0, 0.0, 0.0, 1.0]]))


def test_randomised_pit_is_uniform_for_the_true_distribution() -> None:
    rng = np.random.default_rng(5807)
    mu = rng.uniform(0.5, 4.0, 5000)
    y = rng.poisson(mu)
    pmf = stats.poisson.pmf(np.arange(40)[None, :], mu[:, None])
    pit = pm.randomised_pit(pmf, y, np.random.default_rng(5808))
    assert stats.kstest(pit, "uniform").pvalue > 1e-3


def test_cluster_bootstrap_is_customer_resampling() -> None:
    rng = np.random.default_rng(5809)
    cluster = np.repeat(np.arange(60), rng.integers(1, 5, 60))
    a = rng.normal(1.0, 1.0, cluster.size)
    b = rng.normal(0.8, 1.0, cluster.size)
    est, lo, hi = pm.cluster_bootstrap_mean_difference(a, b, cluster, 400, 0.95,
                                                       np.random.default_rng(5810))
    assert est == pytest.approx(np.mean(a - b)) and lo < est < hi
    # the multinomial weights reproduce an explicit resample of whole customers
    w = np.random.default_rng(5811).multinomial(60, np.full(60, 1 / 60))
    rows = np.concatenate([np.repeat(np.flatnonzero(cluster == c), w[c]) for c in range(60)])
    d_sum = np.bincount(cluster, weights=a - b)
    n_rows = np.bincount(cluster).astype(float)
    assert (w @ d_sum) / (w @ n_rows) == pytest.approx(np.mean((a - b)[rows]))


def test_cr1_matches_statsmodels_cluster_covariance() -> None:
    rng = np.random.default_rng(5812)
    n_c = 80
    cluster = np.repeat(np.arange(n_c), rng.integers(2, 6, n_c))
    X = sm.add_constant(rng.normal(size=(cluster.size, 2)) + rng.normal(size=(n_c, 2))[cluster])
    y = (rng.uniform(size=cluster.size) < 1 / (1 + np.exp(-(X @ [0.2, 0.8, -0.5]
                                                              + rng.normal(size=n_c)[cluster]))))
    ref = sm.GLM(y.astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster})
    beta = ref.params
    p = 1 / (1 + np.exp(-(X @ beta)))
    bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    v = pm.cr1_covariance(bread, X * (y - p)[:, None], cluster)
    np.testing.assert_allclose(v, ref.cov_params(), rtol=1e-6)


def test_clopper_pearson_matches_statsmodels() -> None:
    for k, n in ((0, 20), (7, 20), (20, 20), (431, 500)):
        lo, hi = pm.clopper_pearson(k, n)
        ref = proportion_confint(k, n, alpha=0.05, method="beta")
        assert lo == pytest.approx(ref[0], abs=1e-12) and hi == pytest.approx(ref[1], abs=1e-12)

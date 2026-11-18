"""M11 P3: E2 on the M10 generator, where it is correctly specified within each segment x
channel cell (development seeds). Correctness gates only: parameter recovery, calibration
against realised value, and no aggregate bias against the exact oracle across draws."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.customer_value import contractual as ct
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as orc
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS

W = 12


def _policy(cfg: m10g.EconomicsScenarioConfig) -> PromotionPolicy:
    return PromotionPolicy(cfg.promo_months, tuple(
        c for c, p in zip(CHANNELS, cfg.promo_channels, strict=True) if p))


def _slope_z(pred: np.ndarray, target: np.ndarray) -> float:
    x = pred - pred.mean()
    slope = float(np.sum(x * (target - target.mean())) / np.sum(x * x))
    e = target - target.mean() - slope * x
    return (slope - 1.0) / (math.sqrt(float(np.sum(x * x * e * e))) / float(np.sum(x * x)))


def test_e2_recovers_every_cells_true_parameters() -> None:
    """Joint Wald check over the ``ok`` cells x 2 log parameters (chi-square, 0.999) and a
    Sidak-style per-parameter bound, at both evaluation cutoffs. A small cell's ``v`` is weakly
    identified and its MLE can reach the boundary ``v = 0`` (status ``unstable``, no Wald SE);
    such cells are reported and use the A6 geometric limit or the pooled fallback, never
    dropped."""
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=4821))
    for cutoff in (24, 30):
        fits = ct.predict("E2", sc, cutoff, W, _policy(sc.config)).fits
        zs: list[float] = []
        assert len(fits.used) == len(fits.cells) == 12
        for (s, c), f in fits.cells.items():
            if f.status != "ok":
                assert f.status == "unstable"
                st, used = fits.used[(s, c)]
                # A6: a verified v = 0 boundary uses the geometric limit; otherwise fallback
                assert (st == "ok" and used.boundary is not None) or (
                    st == "fallback" and used is fits.pooled_used)
                continue
            true = np.log([sc.truth.base_hazard(s, c), sc.config.frailty_variance])
            zs.extend(((f.log_params - true) / np.sqrt(np.diag(f.cov_log))).tolist())
        assert len(zs) >= 2 * 10  # at least 10 of 12 cells estimated
        z = np.asarray(zs)
        assert float(np.sum(z * z)) <= stats.chi2.ppf(0.999, z.size)
        assert float(np.max(np.abs(z))) <= 3.5


@pytest.mark.parametrize("cutoff", [24, 30])
def test_e2_is_calibrated_against_realised_value(cutoff: int) -> None:
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=4822))
    pol = _policy(sc.config)
    pred = ct.predict("E2", sc, cutoff, W, pol).frame
    hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", "A", pol)
    r = hold.set_index("customer_id")["realised"].reindex(pred.index).to_numpy()
    p = pred["prediction"].to_numpy()
    d = r - p
    assert abs(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))) <= 4.0
    assert abs(_slope_z(p, r)) <= 4.0


def test_e2_has_no_aggregate_bias_against_the_oracle_across_draws() -> None:
    """Acceptance gate 4 (correct specification): the mean of ``mean(E2 - oracle)`` over
    independent draws is consistent with 0 (|z| <= 4)."""
    diffs = []
    for seed in range(4831, 4839):
        sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=seed))
        pred = ct.predict("E2", sc, 24, W, _policy(sc.config)).frame
        o = orc.m10_oracle(sc, 24, W).frame
        diffs.append(float((pred["prediction"].reindex(o.index) - o["oracle"]).mean()))
    d = np.asarray(diffs)
    assert abs(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))) <= 4.0


def test_v0_case_is_calibrated_after_amendment_a6() -> None:
    """Homogeneous churn (v = 0, the PC2 grid's first point): with A6 the boundary cells use
    the geometric limit; E2 is calibrated against realised value, its boundary bias is bounded
    and its oracle excess falls below half the pre-A6 value."""
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=4841, frailty_variance=0.0))
    pol = _policy(sc.config)
    for cutoff in (24, 30):
        pred = ct.predict("E2", sc, cutoff, W, pol)
        assert pred.fits.boundary_cells
        assert pred.fits.status_counts().get("fallback", 0) < len(pred.fits.boundary_cells)
        hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", "A", pol)
        r = hold.set_index("customer_id")["realised"].reindex(pred.frame.index).to_numpy()
        p = pred.frame["prediction"].to_numpy()
        d = r - p
        assert abs(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))) <= 4.0
        assert abs(_slope_z(p, r)) <= 4.0
    _boundary_bias_gate("E2", {"boundary_exception": False}, range(4842, 4848))


def test_e1_v0_case_is_calibrated_after_amendment_a7() -> None:
    """Homogeneous churn: with A7, E1's ridge cells use the geometric limit; E1 is calibrated
    against realised value, its boundary bias is bounded and its oracle excess falls below
    half the pre-A7 value."""
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
        seed=4851, frailty_variance=0.0))
    pol = _policy(sc.config)
    for cutoff in (24, 30):
        pred = ct.predict("E1", sc, cutoff, W, pol)
        assert pred.fits.boundary_cells
        hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "contractual", "A", pol)
        r = hold.set_index("customer_id")["realised"].reindex(pred.frame.index).to_numpy()
        p = pred.frame["prediction"].to_numpy()
        d = r - p
        assert abs(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))) <= 4.0
        assert abs(_slope_z(p, r)) <= 4.0
    _boundary_bias_gate("E1", {"e1_boundary_exception": False}, range(4852, 4858))


def _boundary_bias_gate(model: str, before: dict[str, bool], seeds: range) -> None:
    """At v = 0 the dispersion parameter is on the boundary, so its MLE is truncated at 0 and
    positively biased: predictions carry a small upward bias that shrinks as ``n^-1/2``
    (about +1% at this scale, +0.6% at 4x; P3 record). Zero aggregate bias is therefore not
    the right gate; instead the relative bias is bounded and the amendment must reduce the
    oracle excess on every draw's average."""
    rel, ex_after, ex_before = [], [], []
    for seed in seeds:
        s2 = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(
            seed=seed, frailty_variance=0.0))
        pol = _policy(s2.config)
        o = orc.m10_oracle(s2, 24, W).frame
        mu = o["oracle"].to_numpy()
        after = ct.predict(model, s2, 24, W, pol).frame["prediction"].reindex(o.index)
        prior = ct.predict(model, s2, 24, W, pol, **before).frame["prediction"].reindex(
            o.index)
        rel.append(float((after.to_numpy() - mu).mean() / mu.mean()))
        ex_after.append(float(np.mean((after.to_numpy() - mu) ** 2)))
        ex_before.append(float(np.mean((prior.to_numpy() - mu) ** 2)))
    assert abs(float(np.mean(rel))) <= 0.025
    assert float(np.mean(ex_after)) < 0.5 * float(np.mean(ex_before))

"""M11 P2: the non-contractual oracle against each variant's own latent truth and realised
outcomes (development seeds). ``E[latent value | F_C] = oracle`` (tower property), so the
latent truth is an unbiased, low-noise check of the oracle; realised value adds purchase and
spend noise."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.data import predictive_ltv_scenarios as g

CUTOFF, W = 24, 12


def _z_mean(d: np.ndarray) -> float:
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))


def _slope_z(pred: np.ndarray, target: np.ndarray, null: float) -> float:
    """``(slope - null) / HC0 SE`` of the OLS regression of ``target`` on ``pred``."""
    x = pred - pred.mean()
    slope = float(np.sum(x * (target - target.mean())) / np.sum(x * x))
    e = target - target.mean() - slope * x
    return (slope - null) / (math.sqrt(float(np.sum(x * x * e * e))) / float(np.sum(x * x)))


def _scenario(variant: str | None, seed: int) -> g.PurchasesScenario:
    cfg = g.PurchasesScenarioConfig(seed=seed)
    return g.generate_purchases_scenario(g.purchases_variant(variant, cfg) if variant else cfg)


@pytest.mark.parametrize("variant,seed", [(None, 4901), ("pareto_nbd_dropout", 4902),
                                          ("correlated_spend", 4903)])
def test_oracle_matches_its_own_latent_truth_and_realised_value(variant: str | None,
                                                                seed: int) -> None:
    sc = _scenario(variant, seed)
    f = po.purchases_oracle(sc, CUTOFF, W).frame
    latent = sc.truth.latent_expected_value(CUTOFF, W).set_index("customer_id").reindex(f.index)
    mu = f["oracle"].to_numpy()
    lat = latent["expected"].to_numpy()
    assert abs(_z_mean(lat - mu)) <= 4.0
    assert abs(_slope_z(mu, lat, 1.0)) <= 4.0
    alive = latent["alive"].to_numpy(dtype=float)
    p = f["p_alive"].to_numpy()
    assert abs(alive.mean() - p.mean()) <= 4.5 * math.sqrt(np.sum(p * (1 - p))) / p.size
    for lo, hi in ((0.0, 0.2), (0.2, 0.8), (0.8, 1.0)):  # reliability of P(alive)
        band = (p > lo) & (p <= hi)
        pb = p[band]
        assert abs(alive[band].mean() - pb.mean()) <= 4.5 * math.sqrt(
            np.sum(pb * (1 - pb))) / pb.size
    spend = sc.truth.expected_spend()[np.searchsorted(sc.truth.customer_ids, f.index)]
    assert abs(_z_mean(spend - f["expected_spend"].to_numpy())) <= 4.0
    hf = lm.holdout_frame(sc, sc.config.n_months, CUTOFF, W, "non_contractual", "A")
    r = hf.set_index("customer_id")["realised"].reindex(f.index).to_numpy()
    assert abs(_z_mean(r - mu)) <= 4.0
    assert abs(_slope_z(mu, r, 1.0)) <= 4.0


def test_a_wrong_dropout_process_is_detected() -> None:
    """Power check: the BG/NBD oracle applied to Pareto/NBD data is visibly miscalibrated for
    P(alive), so the calibration checks above can fail."""
    sc = _scenario("pareto_nbd_dropout", 4904)
    wrong = po.purchases_oracle(replace(sc, config=replace(sc.config, dropout_process="bgnbd")),
                                CUTOFF, W).frame
    alive = sc.truth.alive_at(CUTOFF)[np.searchsorted(sc.truth.customer_ids, wrong.index)]
    assert abs(_z_mean(alive.astype(float) - wrong["p_alive"].to_numpy())) > 8.0


def test_correlated_spend_adjustment_is_the_true_one() -> None:
    """Under the copula, regress (true spend - independent-oracle spend) on (copula - independent
    spend): slope 1 if the copula adjustment is right, 0 if it were unnecessary."""
    sc = _scenario("correlated_spend", 4911)
    cop = po.purchases_oracle(sc, CUTOFF, W).frame
    ind_cfg = replace(sc.config, spend_frequency_correlation=0.0)
    ind = po.purchases_oracle(replace(sc, config=ind_cfg), CUTOFF, W).frame
    true = sc.truth.expected_spend()[np.searchsorted(sc.truth.customer_ids, cop.index)]
    d = cop["expected_spend"].to_numpy() - ind["expected_spend"].to_numpy()
    y = true - ind["expected_spend"].to_numpy()
    assert abs(_slope_z(d, y, 1.0)) <= 4.0
    assert _slope_z(d, y, 0.0) > 5.0

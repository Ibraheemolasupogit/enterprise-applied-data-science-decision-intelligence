"""M11 P5: E4 on the purchases generator, where it is correctly specified within each segment x
channel cell (development seeds). Correctness gates only: per-cell parameter recovery,
calibration against realised value, and no aggregate bias against the exact oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import purchase_oracle as po
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS, SEGMENTS

W = 12


def _slope_z(pred: np.ndarray, target: np.ndarray) -> float:
    x = pred - pred.mean()
    slope = float(np.sum(x * (target - target.mean())) / np.sum(x * x))
    e = target - target.mean() - slope * x
    return (slope - 1.0) / (math.sqrt(float(np.sum(x * x * e * e))) / float(np.sum(x * x)))


def test_e4_recovers_every_cells_true_parameters() -> None:
    sc = g.generate_purchases_scenario(g.PurchasesScenarioConfig(seed=4961))
    cfg = sc.config
    fits = btyd.fit_cells(btyd.purchase_data(sc, 30, W).frame)
    zs: list[float] = []
    for (s, c), bf in fits.bgnbd.items():
        si, ci = SEGMENTS.index(s), CHANNELS.index(c)
        gf = fits.gg[(s, c)]
        for fit, true in ((bf, (cfg.r, cfg.alpha * cfg.segment_alpha_scale[si],
                                cfg.dropout_a, cfg.dropout_b[ci])),
                          (gf, (cfg.gg_p, cfg.gg_q, cfg.gg_gamma[si]))):
            if fit.status != "ok":
                continue
            zs.extend(((fit.log_params - np.log(true))
                       / np.sqrt(np.diag(fit.cov_log))).tolist())
    z = np.asarray(zs)
    assert z.size >= 7 * 10  # at least 10 of 12 cells with both components ok
    assert float(np.sum(z * z)) <= stats.chi2.ppf(0.999, z.size)
    assert float(np.max(np.abs(z))) <= 3.5


@pytest.mark.parametrize("cutoff", [24, 30])
def test_e4_is_calibrated_against_realised_value(cutoff: int) -> None:
    sc = g.generate_purchases_scenario(g.PurchasesScenarioConfig(seed=4962))
    pred = btyd.predict(sc, cutoff, W).frame
    hold = lm.holdout_frame(sc, sc.config.n_months, cutoff, W, "non_contractual", "A")
    r = hold.set_index("customer_id")["realised"].reindex(pred.index).to_numpy()
    p = pred["prediction"].to_numpy()
    d = r - p
    assert abs(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))) <= 4.0
    assert abs(_slope_z(p, r)) <= 4.0


def test_e4_has_no_aggregate_bias_against_the_oracle_across_draws() -> None:
    diffs = []
    for seed in range(4963, 4969):
        sc = g.generate_purchases_scenario(g.PurchasesScenarioConfig(seed=seed))
        pred = btyd.predict(sc, 24, W).frame
        o = po.purchases_oracle(sc, 24, W).frame
        diffs.append(float((pred["prediction"].reindex(o.index) - o["oracle"]).mean()))
    d = np.asarray(diffs)
    assert abs(float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))) <= 4.0

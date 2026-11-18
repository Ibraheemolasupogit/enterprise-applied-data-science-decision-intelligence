"""M11 P2: oracle calibration and the expectation-based oracle gate (amendment 1), on
development seeds, against realised outcomes and the generator's latent truth."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.customer_value import landmarks as lm
from decision_intelligence.customer_value import oracle as o
from decision_intelligence.customer_value.features import PromotionPolicy
from decision_intelligence.data import customer_economics_scenarios as m10g
from decision_intelligence.data import predictive_ltv_scenarios as g
from decision_intelligence.data.customer_economics_scenarios import CHANNELS

CUTOFF, W = 36, 12


def _realised(sc: g.SignalsScenario | m10g.EconomicsScenario, index: pd.Index) -> np.ndarray:
    pol = PromotionPolicy(2, tuple(c for c, p in zip(CHANNELS, sc.config.promo_channels,
                                                       strict=True) if p))
    hf = lm.holdout_frame(sc, sc.config.n_months, CUTOFF, W, "contractual", "A", pol)
    return np.asarray(hf.set_index("customer_id")["realised"].reindex(index), dtype=float)


def _calibrated(mu: np.ndarray, r: np.ndarray) -> None:
    resid = r - mu
    assert abs(resid.mean()) <= 4 * resid.std(ddof=1) / math.sqrt(resid.size)
    x = mu - mu.mean()
    slope = float(np.sum(x * (r - r.mean())) / np.sum(x * x))
    se = float(resid.std(ddof=1) / math.sqrt(np.sum(x * x)))
    assert abs(slope - 1.0) <= 4 * se


@pytest.fixture(scope="module", params=[False, True], ids=["ph", "nph"])
def sig(request: pytest.FixtureRequest) -> g.SignalsScenario:
    cfg = g.SignalsScenarioConfig(seed=4801)
    if request.param:
        cfg = g.signals_variant("non_proportional_hazards", g.SignalsScenarioConfig(seed=4802))
    return g.generate_signals_scenario(cfg)


def test_signals_oracles_are_calibrated(sig: g.SignalsScenario) -> None:
    for tier in ("A", "B"):
        f = o.signals_oracle(sig, CUTOFF, W, tier).frame
        r = _realised(sig, f.index)
        _calibrated(f["oracle"].to_numpy(), r)
        tx = sig.transactions
        for j in (1, 6, 12):  # survival probabilities: P(month C + j paid | F_C)
            paid = tx.loc[tx["billing_month"] == CUTOFF + j, "customer_id"].unique()
            obs = np.isin(f.index.to_numpy(), paid).astype(float)
            p = f[f"paid_{j}"].to_numpy()
            assert abs(obs.mean() - p.mean()) <= 4.5 * math.sqrt(np.sum(p * (1 - p))) / p.size


def test_posteriors_match_the_latent_truth(sig: g.SignalsScenario) -> None:
    tr = sig.truth
    for tier in ("A", "B"):
        res = o.signals_oracle(sig, CUTOFF, W, tier)
        assert res.posterior is not None
        ids = res.frame.index.to_numpy()
        k = tr.latent_class[np.searchsorted(tr.customer_ids, ids)]
        state = tr.states.loc[tr.states["month"] == CUTOFF].set_index("customer_id")[
            "state"].reindex(ids).to_numpy()
        for truth, post in ((k, res.posterior.sum(axis=2)), (state, res.posterior.sum(axis=1))):
            for c in range(3):
                p = post[:, c]
                hit = (truth == c).astype(float)
                assert abs(hit.mean() - p.mean()) <= 4.5 * math.sqrt(np.sum(p * (1 - p))) / p.size


def test_oracle_gate_holds_in_expectation(sig: g.SignalsScenario) -> None:
    """Amendment 1: E[(R - mu_A)^2 - (R - mu_B)^2] = E[(mu_B - mu_A)^2]; the per-customer
    difference 2 (R - mu_B)(mu_B - mu_A) has mean 0. A single seed's sign of D is not
    asserted."""
    a = o.signals_oracle(sig, CUTOFF, W, "A").frame["oracle"]
    b = o.signals_oracle(sig, CUTOFF, W, "B").frame["oracle"].reindex(a.index)
    r = _realised(sig, a.index)
    mu_a, mu_b = a.to_numpy(), b.to_numpy()
    d = (r - mu_a) ** 2 - (r - mu_b) ** 2 - (mu_b - mu_a) ** 2
    assert abs(d.mean()) <= 4 * d.std(ddof=1) / math.sqrt(d.size)
    tower = mu_b - mu_a  # E[mu_B | F_A] = mu_A
    assert abs(tower.mean()) <= 4 * tower.std(ddof=1) / math.sqrt(tower.size)


def test_m10_oracle_is_calibrated() -> None:
    sc = m10g.generate_economics_scenario(m10g.EconomicsScenarioConfig(seed=4803))
    f = o.m10_oracle(sc, CUTOFF - 6, W).frame
    pol = PromotionPolicy(2, ("paid_search", "social"))
    hf = lm.holdout_frame(sc, sc.config.n_months, CUTOFF - 6, W, "contractual", "A", pol)
    r = np.asarray(hf.set_index("customer_id")["realised"].reindex(f.index), dtype=float)
    _calibrated(f["oracle"].to_numpy(), r)

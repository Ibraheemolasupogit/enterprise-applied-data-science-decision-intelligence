"""M11 P7 phase gate: the IPCW Brier score is unbiased under independent censoring
(development seeds only).

Discrete survival times with known heterogeneous hazards, a deliberately imperfect prediction
``S_hat = S^1.3`` and independent censoring on ``0 .. 12``. For each replication the target is the
exact conditional expectation of the complete-data Brier score given the covariates,
``mean_i[S_i(t) (1 - S_hat_i(t))^2 + (1 - S_i(t)) S_hat_i(t)^2]``. The IPCW estimate must agree
with it in expectation (``|z| < 4`` across replications); the naive score that drops subjects
censored before ``t`` must not (it is biased, which is why IPCW is needed).
"""

from __future__ import annotations

import math

import numpy as np

from decision_intelligence.statistics import predictive_metrics as pm

H = 12
REPLICATIONS = 300
N = 800


def _replication(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per t = 1..H: IPCW minus target, naive minus target; and the integrated difference."""
    haz = rng.uniform(0.02, 0.25, N)
    T = rng.geometric(haz) - 1
    C = rng.choice(np.arange(H + 1), size=N, p=np.r_[np.full(H, 0.6 / H), 0.4])
    event = T < C
    time = np.where(event, T, C)
    S = (1 - haz[:, None]) ** np.arange(1, H + 1)[None, :]
    S_hat = S ** 1.3
    target = np.mean(S * (1 - S_hat) ** 2 + (1 - S) * S_hat ** 2, axis=0)
    g = pm.censoring_km(time, event, H)
    ipcw = np.array([pm.ipcw_brier(t, time, event, S_hat[:, t - 1], g) for t in range(1, H + 1)])
    naive = np.empty(H)
    for t in range(1, H + 1):
        known = (time >= t) | event  # status at t known (alive at t, or failed before t)
        status = (time >= t).astype(float)
        naive[t - 1] = np.mean((status[known] - S_hat[known, t - 1]) ** 2)
    ib = pm.integrated_brier(time, event, S_hat) - float(np.mean(target))
    return ipcw - target, naive - target, np.array([ib])


def test_ipcw_brier_is_unbiased_under_independent_censoring() -> None:
    rng = np.random.default_rng(5841)
    runs = [_replication(rng) for _ in range(REPLICATIONS)]
    ipcw = np.array([r[0] for r in runs])
    naive = np.array([r[1] for r in runs])
    integ = np.array([r[2][0] for r in runs])
    z_ipcw = ipcw.mean(axis=0) / (ipcw.std(axis=0, ddof=1) / math.sqrt(REPLICATIONS))
    assert np.all(np.abs(z_ipcw) < 4.0), z_ipcw
    z_int = integ.mean() / (integ.std(ddof=1) / math.sqrt(REPLICATIONS))
    assert abs(z_int) < 4.0
    z_naive = naive.mean(axis=0) / (naive.std(axis=0, ddof=1) / math.sqrt(REPLICATIONS))
    assert np.max(np.abs(z_naive[H // 2:])) > 10.0  # the complete-case score is biased

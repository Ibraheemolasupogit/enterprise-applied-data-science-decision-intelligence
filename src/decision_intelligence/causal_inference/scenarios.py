"""Simulated causal scenarios with known truth, including deliberate failures.

Each scenario generates *both* potential outcomes, so the true estimand is
known exactly, plus the data an analyst would see and the DAG that
generated them. They exist to show when an adjustment strategy works and,
just as importantly, when it fails:

=====================  ======================================================
Scenario               What it demonstrates
=====================  ======================================================
``confounded``         Naive contrast biased; valid backdoor adjustment recovers
                       the effect (heterogeneous effect, ATE ≠ ATT).
``mediator``           Adjusting for a mediator removes the indirect effect:
                       it estimates a (controlled) direct effect, not the total.
``m_bias``             No confounding, yet adjusting for a pre-treatment
                       collider *creates* bias (M-bias).
``positivity``         Near-deterministic assignment: extreme weights, unstable
                       IPW; trimming recovers stability but changes the estimand.
``misspecification``   Double robustness: AIPW is consistent if either nuisance
                       model is right, biased if both are wrong.
``interference``       SUTVA violated by spillovers: the difference in means
                       recovers the direct effect, not the policy effect.
=====================  ======================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special

from decision_intelligence.causal_inference.dag import CausalDAG
from decision_intelligence.causal_inference.estimands import PotentialOutcomes
from decision_intelligence.common.validation import require_positive_int


@dataclass(frozen=True)
class Scenario:
    """Observed data, the full potential outcomes and the generating DAG."""

    name: str
    frame: pd.DataFrame
    """Observed columns: ``treatment``, ``outcome`` and covariates."""
    potential_outcomes: PotentialOutcomes
    dag: CausalDAG | None
    truths: dict[str, float] = field(default_factory=dict)

    @property
    def true_ate(self) -> float:
        return self.potential_outcomes.ate()

    @property
    def true_att(self) -> float:
        return self.potential_outcomes.att()


def _frame(treatment: np.ndarray, outcome: np.ndarray, **covariates: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"treatment": treatment.astype(float), "outcome": outcome, **covariates})


def confounded(n: int, rng: np.random.Generator, *, confounding: float = 1.5) -> Scenario:
    """``X -> T``, ``X -> Y``, ``T -> Y`` with effect ``tau(X) = 2 + X``.

    ``e(X) = sigmoid(confounding * X)``; ``Y(t) = 1 + 2X + t (2 + X) + eps``.
    Treated units have higher ``X`` and therefore a larger effect, so the
    ATT exceeds the ATE (= 2 in expectation).
    """
    n = require_positive_int(n, "n", minimum=10)
    x = rng.normal(size=n)
    t = rng.random(n) < special.expit(confounding * x)
    noise = rng.normal(size=n)
    y0 = 1 + 2 * x + noise
    y1 = y0 + 2 + x
    po = PotentialOutcomes(y0, y1, t)
    return Scenario(
        name="confounded",
        frame=_frame(t, po.observed, x=x),
        potential_outcomes=po,
        dag=CausalDAG([("x", "treatment"), ("x", "outcome"), ("treatment", "outcome")]),
    )


def mediator(n: int, rng: np.random.Generator) -> Scenario:
    """Randomised ``T``; ``T -> M -> Y`` and ``T -> Y`` (direct 1.0, indirect 0.8 × 1.5).

    Total effect = 1.0 + 0.8 × 1.5 = 2.2. Adjusting for ``M`` estimates the
    direct effect (1.0) — a different estimand, not a "better" total effect.
    """
    n = require_positive_int(n, "n", minimum=10)
    t = rng.random(n) < 0.5
    m_noise, y_noise = rng.normal(size=n), rng.normal(size=n)
    m0, m1 = m_noise, 0.8 + m_noise
    y0 = 1.5 * m0 + y_noise
    y1 = 1.0 + 1.5 * m1 + y_noise
    po = PotentialOutcomes(y0, y1, t)
    m = np.where(t, m1, m0)
    return Scenario(
        name="mediator",
        frame=_frame(t, po.observed, mediator=m),
        potential_outcomes=po,
        dag=CausalDAG(
            [("treatment", "mediator"), ("mediator", "outcome"), ("treatment", "outcome")]
        ),
        truths={"direct_effect": 1.0, "indirect_effect": 1.2},
    )


def m_bias(n: int, rng: np.random.Generator) -> Scenario:
    """``U1 -> T``, ``U1 -> C <- U2``, ``U2 -> Y``, ``T -> Y`` (effect 1); ``U1, U2`` unobserved.

    ``T`` and ``Y`` share no common cause, so the naive contrast is unbiased;
    ``C`` is a pre-treatment collider, and adjusting for it opens
    ``T <- U1 -> C <- U2 -> Y``.
    """
    n = require_positive_int(n, "n", minimum=10)
    u1, u2 = rng.normal(size=n), rng.normal(size=n)
    c = u1 + u2 + 0.3 * rng.normal(size=n)
    t = rng.random(n) < special.expit(1.5 * u1)
    noise = rng.normal(size=n)
    y0 = 2 * u2 + noise
    y1 = y0 + 1.0
    po = PotentialOutcomes(y0, y1, t)
    return Scenario(
        name="m_bias",
        frame=_frame(t, po.observed, collider=c),
        potential_outcomes=po,
        dag=CausalDAG(
            [
                ("u1", "treatment"),
                ("u1", "collider"),
                ("u2", "collider"),
                ("u2", "outcome"),
                ("treatment", "outcome"),
            ]
        ),
    )


def positivity(n: int, rng: np.random.Generator, *, strength: float = 4.0) -> Scenario:
    """``X ~ U(-2, 2)``, ``e(X) = sigmoid(strength X)``, effect ``tau(X) = 1 + X²``.

    Large ``strength`` makes treatment nearly deterministic at the edges of
    ``X`` — exactly where the effect is largest — so trimming to the
    overlap region targets a smaller effect than the ATE (``1 + 4/3``).
    ``truths['trimmed_ate_0.1']`` is the true effect among units whose *true*
    propensity lies in ``[0.1, 0.9]``. An analysis that trims on *estimated*
    propensities keeps a slightly different set; score it against
    ``potential_outcomes.individual_effects[kept].mean()`` for that set.
    """
    n = require_positive_int(n, "n", minimum=10)
    x = rng.uniform(-2, 2, size=n)
    e = special.expit(strength * x)
    t = rng.random(n) < e
    noise = rng.normal(size=n)
    y0 = x + noise
    y1 = y0 + 1 + x**2
    po = PotentialOutcomes(y0, y1, t)
    kept = (e >= 0.1) & (e <= 0.9)
    return Scenario(
        name="positivity",
        frame=_frame(t, po.observed, x=x),
        potential_outcomes=po,
        dag=CausalDAG([("x", "treatment"), ("x", "outcome"), ("treatment", "outcome")]),
        truths={
            "population_ate": 1 + 4 / 3,
            "trimmed_ate_0.1": float(np.mean((1 + x**2)[kept])),
        },
    )


def misspecification(n: int, rng: np.random.Generator) -> Scenario:
    """Kang & Schafer (2007)-style design with a known effect of 10.

    True confounders ``z1..z4 ~ N(0, 1)``; ``e = sigmoid(-z1 + 0.5 z2 - 0.25 z3 - 0.1 z4)``;
    ``Y(0) = 210 + 27.4 z1 + 13.7 (z2 + z3 + z4) + eps``, ``Y(1) = Y(0) + 10``.
    The analyst also sees the non-linear transformations ``x1..x4`` of the
    ``z``'s; models in ``x`` are misspecified, models in ``z`` are correct.
    """
    n = require_positive_int(n, "n", minimum=10)
    z = rng.normal(size=(n, 4))
    e = special.expit(-z[:, 0] + 0.5 * z[:, 1] - 0.25 * z[:, 2] - 0.1 * z[:, 3])
    t = rng.random(n) < e
    y0 = 210 + 27.4 * z[:, 0] + 13.7 * (z[:, 1] + z[:, 2] + z[:, 3]) + rng.normal(size=n)
    y1 = y0 + 10.0
    po = PotentialOutcomes(y0, y1, t)
    x1 = np.exp(z[:, 0] / 2)
    x2 = z[:, 1] / (1 + np.exp(z[:, 0])) + 10
    x3 = (z[:, 0] * z[:, 2] / 25 + 0.6) ** 3
    x4 = (z[:, 1] + z[:, 3] + 20) ** 2
    return Scenario(
        name="misspecification",
        frame=_frame(
            t, po.observed,
            z1=z[:, 0], z2=z[:, 1], z3=z[:, 2], z4=z[:, 3], x1=x1, x2=x2, x3=x3, x4=x4,
        ),
        potential_outcomes=po,
        dag=None,
    )


def interference(
    n_clusters: int,
    cluster_size: int,
    rng: np.random.Generator,
    *,
    direct_effect: float = 1.0,
    spillover: float = 2.0,
) -> Scenario:
    """Randomised treatment with spillovers inside clusters (e.g. households).

    ``Y_i = direct · T_i + spillover · (share of *other* cluster members treated) + eps``.
    A unit's outcome depends on others' treatments, so "the" potential
    outcome ``Y_i(t)`` is not defined without fixing everyone else's
    treatment. The recorded potential outcomes hold others' treatments at
    their realised values (so ``ate()`` is the *direct* effect); the
    policy-relevant "treat everyone vs no one" effect is
    ``direct + spillover`` and is reported in ``truths``.
    """
    n_clusters = require_positive_int(n_clusters, "n_clusters", minimum=2)
    cluster_size = require_positive_int(cluster_size, "cluster_size", minimum=2)
    cluster = np.repeat(np.arange(n_clusters), cluster_size)
    t = rng.random(cluster.size) < 0.5
    treated_in_cluster = np.bincount(cluster, weights=t.astype(float))[cluster]
    others_share = (treated_in_cluster - t) / (cluster_size - 1)
    noise = rng.normal(size=cluster.size)
    y0 = spillover * others_share + noise
    y1 = y0 + direct_effect
    po = PotentialOutcomes(y0, y1, t)
    return Scenario(
        name="interference",
        frame=_frame(t, po.observed, cluster=cluster.astype(float)),
        potential_outcomes=po,
        dag=None,
        truths={
            "direct_effect": direct_effect,
            "treat_all_vs_none_effect": direct_effect + spillover,
        },
    )

"""Bayesian A/B analysis with explicit priors, posterior summaries and expected-loss decisions.

**What a Bayesian analysis reports, and what it does not.** With a prior
``pi(theta)`` and the likelihood of the data, the posterior ``p(theta | data)``
answers "given the data *and this prior*, how probable is each effect?":

* a **credible interval** contains ``theta`` with posterior probability 0.95.
  It is *not* a confidence interval: a 95% confidence interval is a procedure
  that covers the fixed true effect in 95% of repeated experiments. With a flat
  prior and a normal likelihood the two intervals coincide numerically, but
  their meanings — and their behaviour under informative priors, optional
  stopping or a misspecified prior — differ (quantified in the M7 evidence);
* ``P(treatment better | data)`` is a statement about ``theta``. It is **not**
  ``1 - p``: a p-value is a probability of data at least as extreme *if the
  effect were zero*. Under a flat prior the one-sided p-value and
  ``1 - P(theta > 0 | data)`` happen to agree for a normal model; under any
  informative prior they do not;
* posterior statements are only as good as the prior. Every function here
  takes the prior explicitly, and the evidence reports prior sensitivity.

Models:

* :func:`beta_binomial_posterior` — conversion in each arm with independent
  Beta priors (exact conjugate posteriors). Every summary is computed by
  one-dimensional numerical integration — deterministic, no Monte Carlo —
  and checked against the closed form of ``P(p_t > p_c)`` (Miller 2015;
  Cook 2005) and Monte Carlo in the tests;
* :func:`normal_effect_posterior` — a normal prior on the effect combined
  with the approximately normal likelihood of the M6 estimate (``estimate``,
  ``SE``). Valid for large samples (CLT), e.g. revenue per user; the SE is
  treated as known.

**Decisions** (:func:`expected_loss_decision`) translate the posterior into
annual business value ``V = theta x scale - cost`` and use the expected loss
of each action (Berger 1985): the expected value forgone by launching when
``V < 0`` or by not launching when ``V > 0``. The rule launches when the
expected loss of launching is below a pre-specified tolerance, stops when the
expected loss of *not* launching is, and otherwise holds for more data.

Sequential use: posterior summaries are computed the same way at any look, and
the posterior remains a valid summary of belief *if the prior is right*. That
does **not** make repeated Bayesian decisions immune to frequentist error
inflation: a "launch when ``P(better) > 0.95``" rule applied at every look
launches a null change far more often than once (M7 evidence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from scipy import integrate, optimize, stats

from decision_intelligence.common.validation import require_probability

_QUAD = {"epsabs": 1e-11, "epsrel": 1e-10, "limit": 200}


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaPrior:
    """``Beta(alpha, beta)`` prior on a conversion rate; ``alpha + beta`` acts as a prior
    sample size."""

    alpha: float
    beta: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta prior parameters must be positive")

    @classmethod
    def uniform(cls) -> BetaPrior:
        return cls(1.0, 1.0, "uniform Beta(1, 1)")

    @classmethod
    def from_mean(cls, mean: float, prior_sample_size: float, label: str = "") -> BetaPrior:
        """Informative prior centred on ``mean`` and worth ``prior_sample_size`` observations."""
        mean = require_probability(mean, "mean")
        if prior_sample_size <= 0:
            raise ValueError("prior_sample_size must be positive")
        return cls(mean * prior_sample_size, (1 - mean) * prior_sample_size,
                   label or f"Beta centred at {mean:g}, weight {prior_sample_size:g}")


@dataclass(frozen=True)
class NormalPrior:
    """``N(mean, sd^2)`` prior on an effect (difference between arms)."""

    mean: float
    sd: float
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("prior mean must be finite")
        if not (self.sd > 0 and math.isfinite(self.sd)):
            raise ValueError("prior sd must be positive and finite (use a large sd for a vague "
                             "prior)")


# ---------------------------------------------------------------------------
# Posterior interface
# ---------------------------------------------------------------------------


class EffectPosterior(Protocol):
    """Posterior distribution of the effect ``theta = treatment - control``."""

    @property
    def mean(self) -> float: ...

    def cdf(self, value: float) -> float: ...

    def shortfall(self, threshold: float) -> float:
        """``E[max(threshold - theta, 0)]``."""
        ...

    def excess(self, threshold: float) -> float:
        """``E[max(theta - threshold, 0)]``."""
        ...


def probability_above(posterior: EffectPosterior, threshold: float = 0.0) -> float:
    """``P(theta > threshold | data)`` — a posterior probability, not a p-value."""
    return 1.0 - posterior.cdf(threshold)


def credible_interval(posterior: EffectPosterior, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed credible interval: ``P(lower <= theta <= upper | data) = level``."""
    level = require_probability(level, "level")
    return (_quantile(posterior, (1 - level) / 2), _quantile(posterior, (1 + level) / 2))


def _quantile(posterior: EffectPosterior, q: float) -> float:
    if isinstance(posterior, NormalEffectPosterior):
        return float(stats.norm.ppf(q, posterior.mean, posterior.sd))
    lo, hi = -1.0, 1.0  # difference of two probabilities
    return float(optimize.brentq(lambda d: posterior.cdf(d) - q, lo, hi, xtol=1e-12))


# ---------------------------------------------------------------------------
# Beta-binomial
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaBinomialPosterior:
    """Independent ``Beta`` posteriors of the control and treatment conversion rates."""

    control_alpha: float
    control_beta: float
    treatment_alpha: float
    treatment_beta: float
    prior_label: str

    def _dist(self, arm: Literal["control", "treatment"]) -> stats.rv_continuous:
        if arm == "control":
            return stats.beta(self.control_alpha, self.control_beta)
        return stats.beta(self.treatment_alpha, self.treatment_beta)

    @property
    def mean(self) -> float:
        return float(self._dist("treatment").mean() - self._dist("control").mean())

    @property
    def mean_control(self) -> float:
        return float(self._dist("control").mean())

    @property
    def mean_treatment(self) -> float:
        return float(self._dist("treatment").mean())

    def _support(self) -> tuple[float, float]:
        """Interval holding all but ~1e-14 of the control posterior (keeps quadrature on the
        peak: with thousands of users the posterior sd is ~0.003 on [0, 1])."""
        control = self._dist("control")
        return float(control.ppf(1e-14)), float(control.isf(1e-14))

    def cdf(self, value: float) -> float:
        """``P(p_t - p_c <= value) = int f_c(x) F_t(x + value) dx``."""
        control, treatment = self._dist("control"), self._dist("treatment")
        s_lo, s_hi = self._support()
        lo, hi = max(s_lo, -value), min(s_hi, 1.0 - value)
        # Control mass where x + value >= 1 has F_t = 1; mass where x + value <= 0 has F_t = 0.
        tail = float(control.sf(max(hi, s_lo))) if value > 0 else 0.0
        if lo >= hi:
            return float(min(1.0, max(0.0, tail)))
        inside, _ = integrate.quad(lambda x: control.pdf(x) * treatment.cdf(x + value), lo, hi,
                                   **_QUAD)
        return float(min(1.0, max(0.0, inside + tail)))

    def _partial_shortfall(self, c: float) -> float:
        """``E[max(c - p_t, 0)]`` for a constant ``c`` (closed form)."""
        if c <= 0:
            return 0.0
        a, b = self.treatment_alpha, self.treatment_beta
        m = a / (a + b)
        if c >= 1:
            return c - m
        return float(c * stats.beta.cdf(c, a, b) - m * stats.beta.cdf(c, a + 1, b))

    def shortfall(self, threshold: float) -> float:
        """``E[max(threshold - (p_t - p_c), 0)] = E_c[ E_t[max(threshold + p_c - p_t, 0)] ]``."""
        control = self._dist("control")
        lo, hi = self._support()
        value, _ = integrate.quad(
            lambda x: control.pdf(x) * self._partial_shortfall(threshold + x), lo, hi, **_QUAD
        )
        return float(max(0.0, value))

    def excess(self, threshold: float) -> float:
        """``E[max(theta - threshold, 0)] = E[theta] - threshold + shortfall(threshold)``."""
        return float(max(0.0, self.mean - threshold + self.shortfall(threshold)))


def beta_binomial_posterior(
    successes_treatment: int,
    n_treatment: int,
    successes_control: int,
    n_control: int,
    *,
    prior_treatment: BetaPrior | None = None,
    prior_control: BetaPrior | None = None,
) -> BetaBinomialPosterior:
    """Conjugate update: ``Beta(a + s, b + n - s)`` in each arm (independent priors)."""
    for s, n in ((successes_treatment, n_treatment), (successes_control, n_control)):
        if n < 1 or not 0 <= s <= n:
            raise ValueError("need 0 <= successes <= n and n >= 1 in each arm")
    pt = prior_treatment or BetaPrior.uniform()
    pc = prior_control or BetaPrior.uniform()
    label = pt.label if pt == pc else f"treatment {pt.label}; control {pc.label}"
    return BetaBinomialPosterior(
        control_alpha=pc.alpha + successes_control,
        control_beta=pc.beta + n_control - successes_control,
        treatment_alpha=pt.alpha + successes_treatment,
        treatment_beta=pt.beta + n_treatment - successes_treatment,
        prior_label=label or "Beta",
    )


# ---------------------------------------------------------------------------
# Normal-normal on the effect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalEffectPosterior:
    mean: float
    sd: float
    prior: NormalPrior
    estimate: float
    standard_error: float

    @property
    def shrinkage(self) -> float:
        """Weight on the prior mean: ``SE^2 / (SE^2 + prior_sd^2)``."""
        return self.standard_error**2 / (self.standard_error**2 + self.prior.sd**2)

    def cdf(self, value: float) -> float:
        return float(stats.norm.cdf(value, self.mean, self.sd))

    def shortfall(self, threshold: float) -> float:
        d = (threshold - self.mean) / self.sd
        return float(self.sd * stats.norm.pdf(d) + (threshold - self.mean) * stats.norm.cdf(d))

    def excess(self, threshold: float) -> float:
        d = (self.mean - threshold) / self.sd
        return float(self.sd * stats.norm.pdf(d) + (self.mean - threshold) * stats.norm.cdf(d))


def normal_effect_posterior(estimate: float, standard_error: float, prior: NormalPrior
                            ) -> NormalEffectPosterior:
    """Precision-weighted combination of the prior and the (normal) likelihood of the estimate."""
    if standard_error <= 0 or not math.isfinite(standard_error):
        raise ValueError("standard_error must be positive and finite")
    if not math.isfinite(estimate):
        raise ValueError("estimate must be finite")
    precision = 1 / prior.sd**2 + 1 / standard_error**2
    mean = (prior.mean / prior.sd**2 + estimate / standard_error**2) / precision
    return NormalEffectPosterior(float(mean), float(math.sqrt(1 / precision)), prior,
                                 float(estimate), float(standard_error))


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

BayesDecision = Literal["LAUNCH", "HOLD", "STOP"]


@dataclass(frozen=True)
class ExpectedLossDecision:
    decision: BayesDecision
    expected_annual_value: float
    """``E[V | data]`` with ``V = theta x scale - annual_cost``."""
    probability_value_positive: float
    """``P(V > 0 | data) = P(theta > break-even | data)``."""
    expected_loss_launch: float
    """``E[max(-V, 0)]``: value expected to be lost by launching."""
    expected_loss_not_launching: float
    """``E[max(V, 0)]``: value expected to be forgone by not launching."""
    break_even_effect: float
    loss_tolerance: float
    reasons: tuple[str, ...]


def expected_loss_decision(
    posterior: EffectPosterior,
    *,
    units_per_year: float,
    value_per_unit_effect: float,
    annual_cost: float,
    loss_tolerance: float,
) -> ExpectedLossDecision:
    """LAUNCH if ``E[loss | launch] <= tolerance``; STOP if ``E[loss | not launch] <=
    tolerance``; else HOLD. Both tolerances are in money and fixed before the data."""
    if units_per_year <= 0 or value_per_unit_effect <= 0 or loss_tolerance < 0:
        raise ValueError("units_per_year and value_per_unit_effect must be positive, "
                         "loss_tolerance non-negative")
    scale = units_per_year * value_per_unit_effect
    break_even = annual_cost / scale
    loss_launch = scale * posterior.shortfall(break_even)
    loss_skip = scale * posterior.excess(break_even)
    value = scale * posterior.mean - annual_cost
    p_positive = 1.0 - posterior.cdf(break_even)
    reasons = [f"E[annual value] = {value:,.0f}; P(value > 0) = {p_positive:.3f}",
               f"expected loss if launched {loss_launch:,.0f}, if not launched {loss_skip:,.0f} "
               f"(tolerance {loss_tolerance:,.0f})"]
    decision: BayesDecision
    if loss_launch <= loss_tolerance and (loss_skip > loss_tolerance or value >= 0):
        decision = "LAUNCH"
    elif loss_skip <= loss_tolerance:
        decision = "STOP"
    else:
        decision = "HOLD"
        reasons.append("both actions risk more than the tolerance: collect more data")
    return ExpectedLossDecision(decision, value, p_positive, loss_launch, loss_skip, break_even,
                                loss_tolerance, tuple(reasons))


__all__ = [
    "BayesDecision",
    "BetaBinomialPosterior",
    "BetaPrior",
    "EffectPosterior",
    "ExpectedLossDecision",
    "NormalEffectPosterior",
    "NormalPrior",
    "beta_binomial_posterior",
    "credible_interval",
    "expected_loss_decision",
    "normal_effect_posterior",
    "probability_above",
]

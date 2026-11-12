"""Pre-specified decision rules: LAUNCH, HOLD, ITERATE or STOP.

Statistical significance answers "is there evidence of *some* effect?";
the business question is "is the effect large enough to act on, without
unacceptable harm?". The rule therefore compares the primary metric's
confidence interval with **two** reference points — zero and the
pre-specified minimum worthwhile effect ``delta`` — and each guardrail's
interval with its non-inferiority margin.

Rules, applied in order (``CI`` = two-sided interval of the benefit, i.e.
the primary effect signed so that positive is good):

1. any validity check FAILs → **HOLD** (the comparison is not interpretable);
2. any guardrail shows harm beyond its margin → **STOP**;
3. ``CI upper < delta`` → **STOP** (a worthwhile benefit is ruled out —
   this, not a large p-value, is evidence of "no useful effect");
4. the pre-specified test and the interval disagree about zero (possible only
   at the significance boundary for non-dual pairs such as the pooled z-test
   and the Newcombe interval; see :mod:`.analysis`) → **HOLD**: a non-zero
   effect is not established, and neither construction is chosen after the
   fact because it gives the preferred answer;
5. ``CI lower > 0``, estimate ``>= delta`` and all guardrails non-inferior → **LAUNCH**;
6. ``CI lower > 0`` otherwise → **ITERATE** (a real but small effect, or a
   guardrail that cannot yet be ruled non-inferior);
7. otherwise → **HOLD**: the interval contains both zero and ``delta``; the
   experiment is inconclusive and needs more data (a non-significant result
   is *not* evidence of no effect).

A non-zero effect is therefore "established" only when the test rejects
(``p <= alpha``) *and* the interval excludes zero.

Guardrails use the same two-sided 95% interval, i.e. a one-sided 97.5%
non-inferiority test. The thresholds are illustrative design choices; their
operating characteristics are simulated in the M6 evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from decision_intelligence.experimentation.analysis import MetricResult
from decision_intelligence.experimentation.design import MetricSpec

Decision = Literal["LAUNCH", "HOLD", "ITERATE", "STOP"]
GuardrailStatus = Literal["PASS", "INCONCLUSIVE", "FAIL"]


@dataclass(frozen=True)
class GuardrailAssessment:
    metric: str
    deterioration: float
    deterioration_ci: tuple[float, float]
    margin: float
    status: GuardrailStatus


def assess_guardrail(result: MetricResult, spec: MetricSpec) -> GuardrailAssessment:
    """Non-inferiority: PASS if the whole interval of deterioration lies below the margin,
    FAIL if it lies wholly above it, INCONCLUSIVE otherwise."""
    if spec.margin is None:
        raise ValueError(f"guardrail {spec.name!r} has no margin")
    sign = 1.0 if spec.direction == "decrease" else -1.0  # deterioration = undesirable change
    lo, hi = sorted((sign * result.effect.ci_lower, sign * result.effect.ci_upper))
    status: GuardrailStatus
    if hi < spec.margin:
        status = "PASS"
    elif lo > spec.margin:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return GuardrailAssessment(spec.name, sign * result.effect.estimate, (lo, hi), spec.margin,
                               status)


@dataclass(frozen=True)
class Recommendation:
    decision: Decision
    reasons: tuple[str, ...]
    benefit: float
    benefit_ci: tuple[float, float]
    minimum_worthwhile_effect: float


def recommend(
    primary: MetricResult,
    *,
    minimum_worthwhile_effect: float,
    direction: Literal["increase", "decrease"] = "increase",
    guardrails: Sequence[GuardrailAssessment] = (),
    validity_failures: Sequence[str] = (),
) -> Recommendation:
    delta = minimum_worthwhile_effect
    if delta <= 0:
        raise ValueError("minimum_worthwhile_effect must be positive")
    sign = 1.0 if direction == "increase" else -1.0
    benefit = sign * primary.effect.estimate
    lo, hi = sorted((sign * primary.effect.ci_lower, sign * primary.effect.ci_upper))
    ci = f"[{lo:+.4f}, {hi:+.4f}]"
    reasons: list[str] = []

    def done(decision: Decision) -> Recommendation:
        return Recommendation(decision, tuple(reasons), benefit, (lo, hi), delta)

    if validity_failures:
        reasons.append("validity checks failed: " + "; ".join(validity_failures))
        reasons.append("the treatment comparison is not interpretable until the cause is found")
        return done("HOLD")
    failed = [g.metric for g in guardrails if g.status == "FAIL"]
    if failed:
        reasons.append(f"guardrail harm beyond margin: {', '.join(failed)}")
        return done("STOP")
    if hi < delta:
        reasons.append(f"benefit CI {ci} lies below the worthwhile effect {delta:g}: a useful "
                       "benefit is ruled out")
        return done("STOP")
    if not primary.test_and_interval_agree:
        reasons.append(
            f"boundary evidence: the pre-specified test (p = {primary.p_value:.4f}, "
            f"{'rejects' if primary.significant else 'does not reject'} at "
            f"{primary.alpha:g}) and the benefit CI {ci} disagree about 0 — a non-zero effect "
            "is not established; more data are needed"
        )
        return done("HOLD")
    inconclusive = [g.metric for g in guardrails if g.status == "INCONCLUSIVE"]
    if lo > 0:
        if benefit >= delta and not inconclusive:
            reasons.append(f"benefit {benefit:+.4f} (CI {ci}) is positive and at least the "
                           f"worthwhile effect {delta:g}; guardrails non-inferior")
            return done("LAUNCH")
        if benefit < delta:
            reasons.append(f"benefit is real (CI {ci} excludes 0) but the estimate is below the "
                           f"worthwhile effect {delta:g}")
        if inconclusive:
            reasons.append(f"guardrails not yet shown non-inferior: {', '.join(inconclusive)}")
        return done("ITERATE")
    reasons.append(f"benefit CI {ci} contains both 0 and the worthwhile effect {delta:g}: "
                   "inconclusive — not evidence of no effect; more data are needed")
    return done("HOLD")


@dataclass(frozen=True)
class BusinessImpact:
    annual_value: float
    annual_value_ci: tuple[float, float]
    break_even_effect: float


def business_impact(
    effect: float,
    ci: tuple[float, float],
    *,
    units_per_year: float,
    value_per_unit_effect: float,
    annual_cost: float,
) -> BusinessImpact:
    """Annual net value ``effect * units * value - cost`` with the effect's interval
    propagated linearly, and the effect at which the change breaks even."""
    if units_per_year <= 0 or value_per_unit_effect <= 0:
        raise ValueError("units_per_year and value_per_unit_effect must be positive")
    scale = units_per_year * value_per_unit_effect
    return BusinessImpact(
        annual_value=effect * scale - annual_cost,
        annual_value_ci=(ci[0] * scale - annual_cost, ci[1] * scale - annual_cost),
        break_even_effect=annual_cost / scale if scale else math.inf,
    )


__all__ = [
    "BusinessImpact",
    "Decision",
    "GuardrailAssessment",
    "GuardrailStatus",
    "Recommendation",
    "assess_guardrail",
    "business_impact",
    "recommend",
]

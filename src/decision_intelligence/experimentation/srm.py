"""Sample ratio mismatch (SRM): detecting that arm sizes deviate from the design.

Under Bernoulli/multinomial assignment with shares ``w``, the observed arm
counts are ``Multinomial(N, w)``. A goodness-of-fit test against ``w`` asks
whether the counts are plausible under the *designed* assignment. A
mismatch means units were lost, added or moved **after** randomisation
(or randomisation itself is broken) — and any such process may be related
to outcomes, so the treatment comparison can no longer be trusted.

**Detection is not diagnosis.** A significant SRM test says *that* the
logged sample is inconsistent with the design, not *why*. Benign and
damaging causes (random logging loss vs. a filter that removes
low-engagement users from one arm) produce the same test result.
:func:`srm_breakdown` localises the imbalance by segment or day, and
comparing the logged sample to the server-side assignment record can
show *which* units are missing — these suggest hypotheses about the cause
that must then be confirmed by engineering investigation.

The default threshold ``p <= 0.001`` is the common industry convention:
the check runs on every experiment, false alarms are costly, and real
SRMs are usually extreme. With complete randomisation (fixed arm sizes) the
multinomial null does not apply to the *assigned* counts; the test is then
applied to the *analysed* sample to detect post-randomisation loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from decision_intelligence.statistics.multiple_testing import adjust_p_values, rejects

SRMMethod = Literal["chi-square", "exact-binomial"]


@dataclass(frozen=True)
class SRMResult:
    arms: tuple[str, ...]
    observed: tuple[int, ...]
    expected_shares: tuple[float, ...]
    expected_counts: tuple[float, ...]
    statistic: float
    df: int
    p_value: float
    method: SRMMethod
    alpha: float
    warnings: tuple[str, ...] = ()

    @property
    def detected(self) -> bool:
        return rejects(self.p_value, self.alpha)

    @property
    def observed_shares(self) -> tuple[float, ...]:
        total = sum(self.observed)
        return tuple(c / total for c in self.observed)

    def interpretation(self) -> str:
        if not self.detected:
            return (
                f"no SRM detected (p = {self.p_value:.3g} > {self.alpha:g}); this does not prove "
                "the absence of post-randomisation loss, only that none large enough to detect"
            )
        return (
            f"SRM detected (p = {self.p_value:.3g} <= {self.alpha:g}): the analysed sample is "
            "inconsistent with the designed allocation. The cause is NOT identified by this test; "
            "do not interpret treatment effects until it is diagnosed"
        )


def srm_test(
    counts: Mapping[str, int],
    expected_shares: Mapping[str, float],
    *,
    alpha: float = 0.001,
    method: SRMMethod = "chi-square",
) -> SRMResult:
    """Pearson chi-square goodness of fit (any number of arms) or exact binomial (two arms)."""
    arms = tuple(expected_shares)
    if set(counts) - set(arms):
        raise ValueError(f"unexpected arms in counts: {sorted(set(counts) - set(arms))}")
    observed = np.array([int(counts.get(a, 0)) for a in arms])
    shares = np.array([float(expected_shares[a]) for a in arms])
    if len(arms) < 2 or (shares <= 0).any() or abs(shares.sum() - 1) > 1e-9:
        raise ValueError("expected shares must be positive for >= 2 arms and sum to 1")
    if (observed < 0).any() or observed.sum() == 0:
        raise ValueError("counts must be non-negative with a positive total")
    total = int(observed.sum())
    expected = total * shares
    warnings: list[str] = []
    if method == "chi-square":
        statistic = float(np.sum((observed - expected) ** 2 / expected))
        df = len(arms) - 1
        p = float(stats.chi2.sf(statistic, df))
        if expected.min() < 5:
            warnings.append("an expected count is below 5: use the exact test")
    elif method == "exact-binomial":
        if len(arms) != 2:
            raise ValueError("the exact binomial test needs exactly two arms")
        result = stats.binomtest(int(observed[0]), total, float(shares[0]))
        statistic, df, p = float(observed[0]), 0, float(result.pvalue)
    else:
        raise ValueError("method must be 'chi-square' or 'exact-binomial'")
    return SRMResult(
        arms=arms,
        observed=tuple(int(c) for c in observed),
        expected_shares=tuple(float(s) for s in shares),
        expected_counts=tuple(float(e) for e in expected),
        statistic=statistic,
        df=df,
        p_value=p,
        method=method,
        alpha=alpha,
        warnings=tuple(warnings),
    )


def srm_from_labels(
    labels: np.ndarray | pd.Series, expected_shares: Mapping[str, float], *, alpha: float = 0.001
) -> SRMResult:
    values = pd.Series(np.asarray(labels)).value_counts()
    return srm_test({str(k): int(v) for k, v in values.items()}, expected_shares, alpha=alpha)


def srm_breakdown(
    frame: pd.DataFrame,
    *,
    arm_column: str,
    by: str,
    expected_shares: Mapping[str, float],
    alpha: float = 0.001,
) -> pd.DataFrame:
    """SRM test within each level of ``by`` with Holm-adjusted p-values.

    Localises *where* the imbalance occurs (a segment, a day, a platform),
    which narrows the hypotheses about its cause. It does not establish the
    cause, and pre-treatment levels (``by`` fixed before assignment) are the
    only meaningful breakdowns.
    """
    rows = []
    for level, group in frame.groupby(by, sort=True):
        result = srm_from_labels(group[arm_column], expected_shares, alpha=alpha)
        row: dict[str, object] = {by: level, "n": int(sum(result.observed))}
        for arm, share in zip(result.arms, result.observed_shares, strict=True):
            row[f"share_{arm}"] = share
        row["p_value"] = result.p_value
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = adjust_p_values(table["p_value"].to_numpy(), method="holm").adjusted_p_values
    table["flagged"] = table["p_holm"] <= alpha  # rejection rule: p <= alpha
    return table


__all__ = ["SRMMethod", "SRMResult", "srm_breakdown", "srm_from_labels", "srm_test"]

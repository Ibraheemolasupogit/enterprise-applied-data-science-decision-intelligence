"""Multiple-testing corrections: identity with statsmodels and defining properties."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from decision_intelligence.statistics.multiple_testing import adjust_p_values

_STATSMODELS = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "benjamini-hochberg": "fdr_bh",
    "benjamini-yekutieli": "fdr_by",
}


@pytest.mark.parametrize("method", list(_STATSMODELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_statsmodels(method: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 0.01, 4), rng.uniform(0, 1, 16)])
    rng.shuffle(p)
    ours = adjust_p_values(p, method=method, alpha=0.05)
    reject, adjusted, _, _ = multipletests(p, alpha=0.05, method=_STATSMODELS[method])
    np.testing.assert_allclose(ours.adjusted_p_values, adjusted, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ours.rejected, reject)


def test_ties_order_and_bounds() -> None:
    p = [0.04, 0.01, 0.04, 0.5, 1.0]
    holm = adjust_p_values(p, method="holm")
    assert holm.adjusted_p_values.tolist() == pytest.approx([0.16, 0.05, 0.16, 1.0, 1.0])
    assert (holm.adjusted_p_values >= np.asarray(p)).all()
    bh = adjust_p_values(p, method="benjamini-hochberg")
    assert bh.controls == "FDR" and holm.controls == "FWER"
    assert (bh.adjusted_p_values <= holm.adjusted_p_values + 1e-15).all()
    single = adjust_p_values([0.03], method="bonferroni")
    assert single.adjusted_p_values[0] == pytest.approx(0.03)


def test_validation() -> None:
    with pytest.raises(ValueError):
        adjust_p_values([0.2, 1.2])
    with pytest.raises(ValueError):
        adjust_p_values([0.2], method="sidak")


def test_rejection_at_exactly_alpha_is_inclusive() -> None:
    """Boundary convention: reject iff p <= alpha (M3 simulation helpers, statsmodels)."""
    from decision_intelligence.statistics.multiple_testing import rejects

    above = float(np.nextafter(0.05, 1.0))
    assert rejects(0.05, 0.05) and not rejects(above, 0.05)
    # Bonferroni with m = 2 maps 0.025 to exactly 0.05 (scaling by 2 is exact in binary).
    result = adjust_p_values([0.025, 0.5], method="bonferroni", alpha=0.05)
    assert result.adjusted_p_values[0] == 0.05
    assert result.rejected.tolist() == [True, False]
    reject, _, _, _ = multipletests([0.025, 0.5], alpha=0.05, method="bonferroni")
    np.testing.assert_array_equal(result.rejected, reject)

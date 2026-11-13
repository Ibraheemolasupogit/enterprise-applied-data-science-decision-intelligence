"""The M9 protocol end to end on a reduced scenario (never the fresh seed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_intelligence.recommendations import advanced_cases as ac
from decision_intelligence.recommendations import cases as m8
from decision_intelligence.recommendations import protocol as pr

SMALL = {"seed": 11, "n_users": 600, "n_items": 150}


@pytest.fixture(scope="module")
def run() -> ac.FullRun:
    return ac.full_run(m8.default_config(**SMALL))


def test_selection_follows_the_frozen_rules(run: ac.FullRun) -> None:
    sel = run.selection
    t = sel.tuning
    assert t.groupby("family")["selected"].sum().eq(1).all()
    assert set(ac.M9_SINGLE) <= set(t["family"])
    assert len(t[t["family"] == "implicit ALS"]) == len(pr.ALS_GRID)
    assert len(t[t["family"] == "blend"]) == 35
    assert sel.best_single_m9 in ac.M9_SINGLE
    assert sel.hybrid in {"blend", *(ac.cell_name(k, w) for k, w in ac.RANKER_CELLS)}
    best = t[t["family"].isin(ac.M9_SINGLE) & t["selected"]]
    assert best.loc[best["validation_ndcg"].idxmax(), "family"] == sel.best_single_m9
    assert 0 < sel.candidate_recall_validation["union"].item() <= 1


def test_views_cover_every_model_and_view(run: ac.FullRun) -> None:
    s = run.views.summary
    assert {"item-kNN", "blend", *ac.M9_SINGLE} <= set(s["model"])
    for col in ("logged_ndcg", "oracle_ndcg", "online_clicks", "IPS", "SNIPS", "DR"):
        assert s[col].notna().all()
    assert (s["logged_lo"] <= s["logged_ndcg"]).all() and (s["logged_ndcg"] <= s["logged_hi"]).all()
    lc = run.views.logged_comparisons
    assert pr.REFERENCE_MODEL not in set(lc["model"])
    assert (lc["adjusted_p_value"] >= lc["p_value"]).all()


def test_candidate_recall_and_cold_start(run: ac.FullRun) -> None:
    rec = ac.test_candidate_recall(run.prepared, run.selection)
    assert (rec.loc["union"] >= rec.drop(index=["union", "candidates_per_user"]).max()).all()
    cs = ac.cold_start(run.prepared, run.selection, run.views, ("item-kNN", "two-tower"))
    assert len(cs.users) == 2 and len(cs.users.columns) == 5


def test_simulated_ab_study(run: ac.FullRun) -> None:
    ab = ac.ab_study(run.prepared, run.selection, run.views, aa_replications=10,
                     coverage_replications=10)
    assert set(ab.policies) == set(pr.AB_DESIGN.arms)
    assert ab.policies["control"] == pr.REFERENCE_MODEL
    # policy selection comes from the validation selection alone (frozen rule)
    assert ab.policies["single_model"] == run.selection.best_single_m9
    assert ab.policies["hybrid"] == run.selection.hybrid
    assert ab.result.srm_passed
    assert len(ab.table) == 2 and set(ab.table["decision"]) <= {"LAUNCH", "ITERATE", "HOLD",
                                                                 "STOP"}
    assert len(ab.prediction) == 2


def test_fixed_configuration_study(run: ac.FullRun) -> None:
    t = ac.fixed_config_study(run.selection, {"seed 12": m8.default_config(
        seed=12, n_users=600, n_items=150)})
    assert set(t["scenario"]) == {"seed 12"} and t["tau_logged_vs_online"].between(-1, 1).all()


def test_full_run_is_reproducible(run: ac.FullRun) -> None:
    again = ac.full_run(m8.default_config(**SMALL))
    pd.testing.assert_frame_equal(run.selection.tuning, again.selection.tuning)
    pd.testing.assert_frame_equal(run.views.summary, again.views.summary)
    assert np.array_equal(run.views.recs["blend"], again.views.recs["blend"])

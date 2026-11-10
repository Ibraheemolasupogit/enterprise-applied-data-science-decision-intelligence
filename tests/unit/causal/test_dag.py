"""DAG logic: d-separation against networkx, path rules, backdoor criterion, roles."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from decision_intelligence.causal_inference.campaign import (
    ADJUSTMENT_STRATEGIES,
    assess_strategy,
    campaign_dag,
)
from decision_intelligence.causal_inference.dag import CausalDAG


def _random_dag(rng: np.random.Generator, n_nodes: int, p_edge: float) -> CausalDAG:
    names = [f"v{i}" for i in range(n_nodes)]
    edges = [
        (names[i], names[j])
        for i, j in itertools.combinations(range(n_nodes), 2)
        if rng.random() < p_edge
    ]
    return CausalDAG(edges, nodes=names)


def _to_networkx(dag: CausalDAG) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(dag.nodes)
    graph.add_edges_from(dag.edges)
    return graph


def test_d_separation_matches_networkx_on_random_dags() -> None:
    rng = np.random.default_rng(1)
    checks = 0
    for _ in range(60):
        dag = _random_dag(rng, int(rng.integers(4, 9)), float(rng.uniform(0.2, 0.6)))
        reference = _to_networkx(dag)
        nodes = sorted(dag.nodes)
        for x, y in itertools.combinations(nodes, 2):
            others = [n for n in nodes if n not in (x, y)]
            for _ in range(3):
                given = {n for n in others if rng.random() < 0.35}
                ours = dag.is_d_separated(x, y, given)
                assert ours == nx.is_d_separator(reference, {x}, {y}, given)
                checks += 1
    assert checks > 1000


def test_path_rule_agrees_with_moralisation() -> None:
    """d-separated  <=>  every path is blocked (two independent implementations)."""
    rng = np.random.default_rng(2)
    for _ in range(40):
        dag = _random_dag(rng, 6, 0.45)
        nodes = sorted(dag.nodes)
        for x, y in itertools.combinations(nodes, 2):
            given = {n for n in nodes if n not in (x, y) and rng.random() < 0.3}
            all_blocked = all(dag.path_report(p, given).blocked for p in dag.paths(x, y))
            assert all_blocked == dag.is_d_separated(x, y, given)


def test_chain_fork_and_collider_rules() -> None:
    chain = CausalDAG([("a", "b"), ("b", "c")])
    assert not chain.is_d_separated("a", "c") and chain.is_d_separated("a", "c", {"b"})
    fork = CausalDAG([("b", "a"), ("b", "c")])
    assert not fork.is_d_separated("a", "c") and fork.is_d_separated("a", "c", {"b"})
    collider = CausalDAG([("a", "b"), ("c", "b"), ("b", "d")])
    assert collider.is_d_separated("a", "c")
    assert not collider.is_d_separated("a", "c", {"b"})
    assert not collider.is_d_separated("a", "c", {"d"})  # descendant of a collider opens it
    report = collider.path_report(("a", "b", "c"))
    assert report.colliders == ("b",) and report.blocked


def test_structure_validation() -> None:
    with pytest.raises(ValueError, match="cycle"):
        CausalDAG([("a", "b"), ("b", "c"), ("c", "a")])
    with pytest.raises(ValueError, match="self-loop"):
        CausalDAG([("a", "a")])
    dag = CausalDAG([("a", "b")])
    with pytest.raises(ValueError, match="unknown node"):
        dag.parents("z")
    with pytest.raises(ValueError, match="disjoint"):
        dag.is_d_separated("a", "a")


def test_backdoor_classic_graphs() -> None:
    confounded = CausalDAG([("x", "t"), ("x", "y"), ("t", "y")])
    assert not confounded.satisfies_backdoor("t", "y", [])
    assert confounded.satisfies_backdoor("t", "y", ["x"])
    m_bias = CausalDAG([("u1", "t"), ("u1", "c"), ("u2", "c"), ("u2", "y"), ("t", "y")])
    assert m_bias.satisfies_backdoor("t", "y", [])
    assert not m_bias.satisfies_backdoor("t", "y", ["c"])
    assert m_bias.satisfies_backdoor("t", "y", ["c", "u1"])
    mediated = CausalDAG([("t", "m"), ("m", "y"), ("t", "y")])
    assessment = mediated.assess_adjustment_set("t", "y", ["m"])
    assert not assessment.valid and assessment.post_treatment_variables == ("m",)


def test_backdoor_matches_networkx_d_separation_in_backdoor_graph() -> None:
    rng = np.random.default_rng(3)
    for _ in range(60):
        dag = _random_dag(rng, 7, 0.4)
        t, y = "v0", "v6"
        reference = _to_networkx(dag.without_outgoing_edges(t))
        descendants = dag.descendants(t)
        others = sorted(dag.nodes - {t, y})
        for size in range(3):
            for subset in itertools.combinations(others, size):
                expected = not (set(subset) & descendants) and nx.is_d_separator(
                    reference, {t}, {y}, set(subset)
                )
                assert dag.satisfies_backdoor(t, y, subset) == expected
                assert dag.assess_adjustment_set(t, y, subset).valid == expected


def test_minimal_adjustment_sets() -> None:
    dag = CausalDAG(
        [("z1", "x"), ("z1", "z3"), ("z2", "z3"), ("z2", "y"), ("z3", "x"), ("z3", "y"),
         ("x", "y")]
    )  # Pearl's classic example: {z3} alone opens the z1 -> z3 <- z2 collider path
    minimal = dag.valid_adjustment_sets("x", "y", ["z1", "z2", "z3"])
    assert set(minimal) == {frozenset({"z1", "z3"}), frozenset({"z2", "z3"})}
    assert not dag.satisfies_backdoor("x", "y", ["z3"])


def test_roles() -> None:
    dag = CausalDAG(
        [("c", "t"), ("c", "y"), ("t", "m"), ("m", "y"), ("t", "s"), ("y", "s")]
    )
    roles = dag.roles("t", "y")
    assert roles["confounders"] == {"c"}
    assert roles["mediators"] == {"m"}
    assert {"m", "s"} <= roles["post_treatment"]
    assert "s" in roles["colliders_on_treatment_outcome_paths"]


def test_campaign_dag_identification() -> None:
    dag = campaign_dag()
    measured = [
        "segment", "latent_value", "price_sensitivity", "household_income",
        "acquisition_channel", "region", "signup_date", "billing_transactions",
    ]
    assert dag.valid_adjustment_sets("treated", "retained_90d", measured) == [
        frozenset({"latent_value"})
    ]
    roles = dag.roles("treated", "retained_90d")
    assert {"latent_value", "segment"} <= roles["confounders"]
    assert "churn_time" in roles["mediators"]
    assert "billing_transactions" in roles["post_treatment"]
    # Testable implications used by the falsification test and negative control.
    assert dag.is_d_separated("treated", "segment", {"latent_value"})
    assert dag.is_d_separated("treated", "price_sensitivity", {"latent_value"})
    assert dag.is_d_separated("treated", "household_income", {"latent_value"})
    assert not dag.is_d_separated("treated", "household_income", {"segment", "price_sensitivity"})


@pytest.mark.parametrize(
    ("strategy", "valid"),
    [
        ("none (naive)", False),
        ("segment only", False),
        ("regime B: observed proxies", False),
        ("regime A: latent_value", True),
        ("regime A: latent_value + segment", True),
        ("INVALID: regime A + post-treatment billing count", False),
    ],
)
def test_campaign_strategies_assessed_by_backdoor_criterion(strategy: str, valid: bool) -> None:
    assessment = assess_strategy(ADJUSTMENT_STRATEGIES[strategy])
    assert assessment.valid is valid
    if strategy.startswith("INVALID"):
        assert assessment.post_treatment_variables == ("billing_transactions",)
    if strategy.startswith("regime B"):
        assert any("latent_value" in p.nodes for p in assessment.open_backdoor_paths)


def test_campaign_billing_node_parents_match_the_generator() -> None:
    """Regression test: the billing *count* depends only on signup and churn dates
    (cp.billing_periods reproduces the pipeline exactly from those two inputs);
    price sensitivity affects billed amounts, not the count."""
    assert campaign_dag().parents("billing_transactions") == {"churn_time", "signup_date"}

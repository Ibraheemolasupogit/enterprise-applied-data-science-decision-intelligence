"""Directed acyclic graphs: d-separation, causal paths and the backdoor criterion.

A causal DAG encodes qualitative assumptions — which variables may
directly cause which. It does not come from the data; the data can only
falsify some of its testable implications (conditional independencies).
Everything here is graph logic on the *assumed* DAG.

Definitions used (Pearl 2009):

* A path between ``a`` and ``b`` is a sequence of distinct nodes joined by
  edges in either direction.
* A node is a **collider** on a path if both adjacent edges point into it
  (``-> c <-``).
* A path is **blocked** by a set ``Z`` if it contains a non-collider in
  ``Z``, or a collider that is not in ``Z`` and has no descendant in ``Z``.
* ``X`` and ``Y`` are **d-separated** by ``Z`` if every path between them is
  blocked. Implemented by the equivalent moralised-ancestral-graph test
  (Lauritzen et al. 1990), and cross-checked by explicit path enumeration.
* **Backdoor criterion** for the effect of ``T`` on ``Y``: ``Z`` contains no
  descendant of ``T`` and blocks every path between ``T`` and ``Y`` that
  starts with an arrow into ``T``. Equivalently, ``Z`` d-separates ``T`` and
  ``Y`` in the graph with ``T``'s outgoing edges removed. If it holds,
  ``E[Y(t)] = E_Z[ E[Y | T=t, Z] ]``.

Role vocabulary relative to a treatment ``T`` and outcome ``Y``:

* *confounder* — a common cause: an ancestor of ``T`` that also reaches ``Y``
  by a directed path not passing through ``T``;
* *mediator* — on a directed path ``T -> ... -> Y``;
* *post-treatment* — any descendant of ``T`` (never a valid adjustment
  variable for the total effect);
* *collider* — a node where two arrowheads meet on some ``T``–``Y`` path;
  conditioning on it (or a descendant) opens that path.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PathReport:
    """One path between treatment and outcome and its status given an adjustment set."""

    nodes: tuple[str, ...]
    is_backdoor: bool
    is_causal: bool
    colliders: tuple[str, ...]
    blocked: bool
    reason: str

    def render(self, graph: CausalDAG) -> str:
        parts = [self.nodes[0]]
        for a, b in itertools.pairwise(self.nodes):
            parts.append("->" if graph.has_edge(a, b) else "<-")
            parts.append(b)
        return " ".join(parts)


@dataclass(frozen=True)
class AdjustmentAssessment:
    """Whether an adjustment set satisfies the backdoor criterion, and why."""

    adjustment_set: frozenset[str]
    valid: bool
    post_treatment_variables: tuple[str, ...]
    open_backdoor_paths: tuple[PathReport, ...]
    explanation: str


class CausalDAG:
    """An immutable directed acyclic graph over named variables."""

    def __init__(self, edges: Iterable[tuple[str, str]], nodes: Iterable[str] = ()) -> None:
        self._nodes: set[str] = set(nodes)
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}
        for node in self._nodes:
            self._children.setdefault(node, set())
            self._parents.setdefault(node, set())
        for parent, child in edges:
            if parent == child:
                raise ValueError(f"self-loop on {parent!r} is not allowed in a DAG")
            for node in (parent, child):
                self._nodes.add(node)
                self._children.setdefault(node, set())
                self._parents.setdefault(node, set())
            self._children[parent].add(child)
            self._parents[child].add(parent)
        cycle = self._find_cycle()
        if cycle:
            raise ValueError(f"graph contains a directed cycle: {' -> '.join(cycle)}")

    # -- basic structure -------------------------------------------------

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._nodes)

    @property
    def edges(self) -> frozenset[tuple[str, str]]:
        return frozenset((p, c) for p, children in self._children.items() for c in children)

    def has_edge(self, parent: str, child: str) -> bool:
        return child in self._children.get(parent, set())

    def parents(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(self._parents[node])

    def children(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(self._children[node])

    def ancestors(self, nodes: str | Iterable[str]) -> frozenset[str]:
        """Strict ancestors of ``nodes`` (excluding the nodes themselves)."""
        return self._closure(nodes, self._parents)

    def descendants(self, nodes: str | Iterable[str]) -> frozenset[str]:
        """Strict descendants of ``nodes`` (excluding the nodes themselves)."""
        return self._closure(nodes, self._children)

    def without_outgoing_edges(self, node: str) -> CausalDAG:
        self._require(node)
        return CausalDAG(
            [(p, c) for p, c in self.edges if p != node], nodes=self._nodes
        )

    # -- d-separation ----------------------------------------------------

    def is_d_separated(
        self, x: str | Iterable[str], y: str | Iterable[str], given: Iterable[str] = ()
    ) -> bool:
        """``x`` and ``y`` are d-separated by ``given`` (moralised ancestral graph test)."""
        xs, ys, zs = self._as_set(x), self._as_set(y), self._as_set(given)
        if xs & ys:
            raise ValueError("x and y must be disjoint")
        if (xs | ys) & zs:
            raise ValueError("conditioning set must not contain x or y")
        relevant = xs | ys | zs
        relevant |= self.ancestors(relevant)
        neighbours: dict[str, set[str]] = {node: set() for node in relevant}
        for child in relevant:
            parents = self._parents[child] & relevant
            for parent in parents:
                neighbours[parent].add(child)
                neighbours[child].add(parent)
            for a, b in itertools.combinations(parents, 2):  # moralise: marry parents
                neighbours[a].add(b)
                neighbours[b].add(a)
        frontier = list(xs)
        seen = set(xs)
        while frontier:
            node = frontier.pop()
            for neighbour in neighbours[node]:
                if neighbour in zs or neighbour in seen:
                    continue
                if neighbour in ys:
                    return False
                seen.add(neighbour)
                frontier.append(neighbour)
        return True

    # -- paths -------------------------------------------------------------

    def paths(self, source: str, target: str) -> list[tuple[str, ...]]:
        """All simple paths between two nodes, ignoring edge direction."""
        self._require(source)
        self._require(target)
        results: list[tuple[str, ...]] = []

        def extend(path: list[str]) -> None:
            last = path[-1]
            for neighbour in sorted(self._children[last] | self._parents[last]):
                if neighbour in path:
                    continue
                if neighbour == target:
                    results.append((*path, target))
                else:
                    extend([*path, neighbour])

        extend([source])
        return results

    def path_report(self, path: Sequence[str], given: Iterable[str] = ()) -> PathReport:
        """Classify one path and decide whether ``given`` blocks it (explicit path rule)."""
        zs = self._as_set(given)
        nodes = tuple(path)
        colliders: list[str] = []
        blocked_reason = ""
        for index in range(1, len(nodes) - 1):
            previous, node, following = nodes[index - 1], nodes[index], nodes[index + 1]
            is_collider = self.has_edge(previous, node) and self.has_edge(following, node)
            if is_collider:
                colliders.append(node)
                opened = node in zs or bool(self.descendants(node) & zs)
                if not opened and not blocked_reason:
                    blocked_reason = f"collider {node!r} not conditioned on"
            elif node in zs and not blocked_reason:
                blocked_reason = f"non-collider {node!r} conditioned on"
        is_causal = all(self.has_edge(a, b) for a, b in itertools.pairwise(nodes))
        return PathReport(
            nodes=nodes,
            is_backdoor=len(nodes) > 1 and self.has_edge(nodes[1], nodes[0]),
            is_causal=is_causal,
            colliders=tuple(colliders),
            blocked=bool(blocked_reason),
            reason=blocked_reason or "open",
        )

    # -- identification ----------------------------------------------------

    def assess_adjustment_set(
        self, treatment: str, outcome: str, adjustment_set: Iterable[str]
    ) -> AdjustmentAssessment:
        """Check the backdoor criterion for ``adjustment_set`` and explain any failure."""
        self._require(treatment)
        self._require(outcome)
        zs = frozenset(self._as_set(adjustment_set))
        if {treatment, outcome} & zs:
            raise ValueError("adjustment set must not contain the treatment or outcome")
        post_treatment = tuple(sorted(zs & self.descendants(treatment)))
        open_backdoor = tuple(
            report
            for report in (self.path_report(p, zs) for p in self.paths(treatment, outcome))
            if report.is_backdoor and not report.blocked
        )
        valid = not post_treatment and not open_backdoor
        if valid:
            explanation = "satisfies the backdoor criterion"
        else:
            problems = []
            if post_treatment:
                problems.append(f"contains post-treatment variable(s) {list(post_treatment)}")
            if open_backdoor:
                problems.append(
                    f"leaves {len(open_backdoor)} backdoor path(s) open, e.g. "
                    f"{open_backdoor[0].render(self)}"
                )
            explanation = "; ".join(problems)
        return AdjustmentAssessment(
            adjustment_set=zs,
            valid=valid,
            post_treatment_variables=post_treatment,
            open_backdoor_paths=open_backdoor,
            explanation=explanation,
        )

    def satisfies_backdoor(
        self, treatment: str, outcome: str, adjustment_set: Iterable[str]
    ) -> bool:
        """Backdoor criterion via d-separation in the graph without ``treatment``'s out-edges."""
        zs = self._as_set(adjustment_set)
        if zs & self.descendants(treatment):
            return False
        return self.without_outgoing_edges(treatment).is_d_separated(treatment, outcome, zs)

    def valid_adjustment_sets(
        self,
        treatment: str,
        outcome: str,
        candidates: Iterable[str],
        *,
        minimal_only: bool = True,
    ) -> list[frozenset[str]]:
        """Backdoor-valid subsets of ``candidates`` (the variables actually measured).

        Exhaustive search, so intended for the small graphs used in analysis
        plans. With ``minimal_only`` only sets with no valid proper subset
        are returned. An empty result means the effect is *not identified by
        backdoor adjustment* with the measured variables.
        """
        pool = sorted(self._as_set(candidates) - {treatment, outcome})
        if len(pool) > 16:
            raise ValueError("too many candidate variables for exhaustive search (max 16)")
        valid = [
            frozenset(subset)
            for size in range(len(pool) + 1)
            for subset in itertools.combinations(pool, size)
            if self.satisfies_backdoor(treatment, outcome, subset)
        ]
        if not minimal_only:
            return valid
        return [s for s in valid if not any(other < s for other in valid)]

    def roles(self, treatment: str, outcome: str) -> dict[str, frozenset[str]]:
        """Classify nodes relative to the effect of ``treatment`` on ``outcome``."""
        self._require(treatment)
        self._require(outcome)
        post = self.descendants(treatment)
        reaches_outcome_avoiding_t = self._without(treatment).ancestors(outcome) if (
            outcome != treatment
        ) else frozenset()
        confounders = frozenset(
            node
            for node in self.ancestors(treatment)
            if node in reaches_outcome_avoiding_t
        )
        mediators = frozenset(post & self.ancestors(outcome))
        colliders: set[str] = set()
        for path in self.paths(treatment, outcome):
            colliders.update(self.path_report(path).colliders)
        return {
            "confounders": confounders,
            "mediators": mediators,
            "post_treatment": frozenset(post - {outcome}),
            "colliders_on_treatment_outcome_paths": frozenset(colliders),
        }

    def implied_independencies(
        self, pairs: Iterable[tuple[str, str]], given: Iterable[str]
    ) -> dict[tuple[str, str], bool]:
        """For each pair, whether the DAG implies independence given ``given``."""
        zs = self._as_set(given)
        return {(a, b): self.is_d_separated(a, b, zs - {a, b}) for a, b in pairs}

    # -- internals ---------------------------------------------------------

    def _without(self, node: str) -> CausalDAG:
        return CausalDAG(
            [(p, c) for p, c in self.edges if node not in (p, c)], nodes=self._nodes - {node}
        )

    def _closure(self, nodes: str | Iterable[str], step: dict[str, set[str]]) -> frozenset[str]:
        start = self._as_set(nodes)
        result: set[str] = set()
        frontier = list(start)
        while frontier:
            node = frontier.pop()
            for nxt in step[node]:
                if nxt not in result:
                    result.add(nxt)
                    frontier.append(nxt)
        # In a DAG no node is its own ancestor/descendant; for a *set* of nodes the
        # closure may include members that are ancestors/descendants of other members.
        return frozenset(result)

    def _as_set(self, nodes: str | Iterable[str]) -> set[str]:
        result = {nodes} if isinstance(nodes, str) else set(nodes)
        for node in result:
            self._require(node)
        return result

    def _require(self, node: str) -> None:
        if node not in self._nodes:
            raise ValueError(f"unknown node {node!r}")

    def _find_cycle(self) -> list[str]:
        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(node: str) -> list[str]:
            state[node] = 1
            stack.append(node)
            for child in sorted(self._children[node]):
                if state.get(child) == 1:
                    return [*stack[stack.index(child) :], child]
                if child not in state:
                    cycle = visit(child)
                    if cycle:
                        return cycle
            stack.pop()
            state[node] = 2
            return []

        for node in sorted(self._nodes):
            if node not in state:
                cycle = visit(node)
                if cycle:
                    return cycle
        return []

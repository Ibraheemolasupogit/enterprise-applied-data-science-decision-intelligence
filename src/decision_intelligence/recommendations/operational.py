"""Operational benchmarks: fit time, latency, throughput, model size, memory (Milestone 9).

**Data.** Operational cost depends on shape and sparsity, not on preference truth, so the scale
tiers use :func:`scale_click_matrix`: a seeded sparse click-count matrix with a Zipf-like item
popularity and Poisson user activity (about 20 clicked items per user). It is generated
directly in sparse form, so the largest tier never materialises a dense user x item matrix.

**Preflight memory check.** Before a workload runs, :func:`planned_bytes` estimates its largest
simultaneous dense allocations (documented per model below). If the estimate exceeds the budget
(:data:`~decision_intelligence.recommendations.protocol.MEMORY_BUDGET_BYTES`) the workload is
**skipped** and reported with its estimate — never attempted.

**Measurements.** Fit wall time and peak traced memory during the fit (``tracemalloc``, which
NumPy reports its buffers to; it adds some overhead to the fit time); model-state size (bytes
of the arrays a model keeps for scoring — reproducible); single-user request latency (score all
items + top-K, ``LATENCY_REQUESTS`` requests: p50 and p95) and batch throughput (users per
second for a batch of 1,000). Times depend on the machine and its load: they are indicative,
not reproducible numbers.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations.factorisation import fit_als, fit_bpr, fit_ease

BYTES = 8


def scale_click_matrix(n_users: int, n_items: int, *, mean_items_per_user: float = 20.0,
                       zipf: float = 1.0, seed: int = 0) -> sparse.csr_matrix:
    """Seeded sparse click counts with Zipf-like item popularity (no preference truth)."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_items + 1) ** zipf
    weights = weights[rng.permutation(n_items)]
    weights /= weights.sum()
    n_per = np.minimum(np.maximum(rng.poisson(mean_items_per_user, n_users), 1), n_items)
    rows, cols = [], []
    for u, n in enumerate(n_per):
        items = rng.choice(n_items, size=int(n), replace=False, p=weights)
        rows.append(np.full(items.size, u))
        cols.append(items)
    r, c = np.concatenate(rows), np.concatenate(cols)
    counts = 1.0 + rng.poisson(0.5, r.size)
    return sparse.csr_matrix((counts, (r, c)), shape=(n_users, n_items))


def planned_bytes(model: str, n_users: int, n_items: int, factors: int = 32,
                  batch: int = 1000) -> int:
    """Largest simultaneous dense allocations planned for fitting and batch scoring.

    * dense-input M8 models (popularity excepted): the dense user x item click matrix
      (``InteractionData``) plus the scoring batch; item-kNN adds three item x item arrays
      (similarity, sort order, pruned copy); PureSVD adds the SVD's ``U`` (user x item) and
      the projected scores;
    * EASE: three item x item arrays (Gram, inverse, weights) — sparse input;
    * ALS / BPR: factor matrices, ``(users + items) x factors``, and the scoring batch;
    * all models: a ``batch x items`` score block.
    """
    ui = n_users * n_items * BYTES
    ii = n_items * n_items * BYTES
    score_block = batch * n_items * BYTES
    factors_b = (n_users + n_items) * factors * BYTES
    plans = {
        "global popularity": n_items * BYTES + score_block,
        "item-kNN (dense M8)": ui + 3 * ii + score_block,
        "PureSVD (dense M8)": 2 * ui + score_block,
        "EASE": 3 * ii + score_block,
        "implicit ALS": 2 * factors_b + score_block,
        "BPR": 2 * factors_b + score_block,
    }
    if model not in plans:
        raise ValueError(f"unknown model {model!r}")
    return int(plans[model])


@dataclass(frozen=True)
class BenchmarkRow:
    model: str
    n_users: int
    n_items: int
    status: str
    planned_bytes: int
    fit_seconds: float = float("nan")
    peak_traced_bytes: float = float("nan")
    model_bytes: float = float("nan")
    latency_p50_ms: float = float("nan")
    latency_p95_ms: float = float("nan")
    throughput_users_per_s: float = float("nan")


Scorer = Callable[[np.ndarray], FloatArray]


def _fit_model(name: str, x: sparse.csr_matrix, factors: int, seed: int
               ) -> tuple[Scorer, int]:
    """Fit on sparse counts; return a batch scorer and the model-state size in bytes."""
    binary = x.copy()
    binary.data[:] = 1.0
    if name == "global popularity":
        pop = np.asarray(binary.sum(axis=0)).ravel()
        return (lambda users: np.broadcast_to(pop, (users.size, pop.size)).copy()), pop.nbytes
    if name == "implicit ALS":
        f = fit_als(x, factors=factors, reg=1.0, alpha=20.0, iterations=15, seed=seed)
        return (lambda users: f.user[users] @ f.item.T), f.user.nbytes + f.item.nbytes
    if name == "BPR":
        b = fit_bpr(binary, factors=factors, reg=0.01, learning_rate=0.05, epochs=30, seed=seed)
        p = b.params
        return ((lambda users: p.user[users] @ p.item.T + p.bias[None, :]),
                p.user.nbytes + p.item.nbytes + p.bias.nbytes)
    if name == "EASE":
        w = fit_ease(binary, 500.0)
        return (lambda users: np.asarray(binary[users] @ w)), w.nbytes + binary.data.nbytes
    if name == "item-kNN (dense M8)":
        dense = binary.toarray()
        norms = np.linalg.norm(dense, axis=0)
        sim = (dense.T @ dense) / (norms[:, None] * norms[None, :] + 10.0)
        np.fill_diagonal(sim, 0.0)
        order = np.argsort(-sim, axis=0, kind="stable")[:100]
        pruned = np.zeros_like(sim)
        cols = np.arange(sim.shape[1])[None, :]
        pruned[order, cols] = sim[order, cols]
        return (lambda users: dense[users] @ pruned), dense.nbytes + pruned.nbytes
    if name == "PureSVD (dense M8)":
        dense = binary.toarray()
        _, _, vt = np.linalg.svd(dense, full_matrices=False)
        v = vt[:factors].T
        return (lambda users: (dense[users] @ v) @ v.T), dense.nbytes + v.nbytes
    raise ValueError(f"unknown model {name!r}")


def _top_k(scores: FloatArray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    return np.asarray(part[rows, np.argsort(-scores[rows, part], axis=1, kind="stable")])


def benchmark_model(name: str, x: sparse.csr_matrix, *, budget: int, factors: int = 32,
                    k: int = 10, requests: int = 200, seed: int = 0) -> BenchmarkRow:
    n_u, n_i = x.shape
    plan = planned_bytes(name, n_u, n_i, factors)
    if plan > budget:
        return BenchmarkRow(name, n_u, n_i, f"SKIPPED: planned {plan / 1e9:.2f} GB > budget "
                            f"{budget / 1e9:.2f} GB", plan)
    tracemalloc.start()
    start = time.perf_counter()
    scorer, model_bytes = _fit_model(name, x, factors, seed)
    fit_s = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rng = np.random.default_rng(seed)
    lat = []
    for u in rng.integers(0, n_u, requests):
        t0 = time.perf_counter()
        _top_k(np.atleast_2d(scorer(np.array([u]))), k)
        lat.append((time.perf_counter() - t0) * 1000)
    batch = rng.integers(0, n_u, min(1000, n_u))
    t0 = time.perf_counter()
    _top_k(np.atleast_2d(scorer(batch)), k)
    thr = batch.size / (time.perf_counter() - t0)
    return BenchmarkRow(name, n_u, n_i, "RUN", plan, fit_s, float(peak), float(model_bytes),
                        float(np.percentile(lat, 50)), float(np.percentile(lat, 95)), thr)


@dataclass(frozen=True)
class RetrievalRow:
    n_items: int
    method: str
    n_clusters: int
    n_probe: int
    recall_of_exact: float
    items_scored_per_query: float
    latency_p50_ms: float


def retrieval_benchmark(queries: FloatArray, items: FloatArray, *, k: int,
                        clusters: tuple[int, ...], probes: tuple[int, ...], seed: int = 0
                        ) -> list[RetrievalRow]:
    """Exact search vs a k-means inverted-file index (recall of the exact top-k, latency)."""
    from decision_intelligence.recommendations.retrieval import (
        KMeansIndex,
        exact_top_k,
        recall_of_exact,
    )

    allowed = np.ones((queries.shape[0], items.shape[0]), dtype=bool)
    lat = []
    for r in range(queries.shape[0]):
        t0 = time.perf_counter()
        exact_top_k(queries[r:r + 1], items, allowed[r:r + 1], k)
        lat.append((time.perf_counter() - t0) * 1000)
    exact = exact_top_k(queries, items, allowed, k)
    rows = [RetrievalRow(items.shape[0], "exact", 0, 0, 1.0, float(items.shape[0]),
                         float(np.percentile(lat, 50)))]
    for c in clusters:
        if c > items.shape[0]:
            continue
        index = KMeansIndex(c, seed=seed).build(items)
        for p in probes:
            if p > c:
                continue
            lat = []
            approx = np.empty_like(exact)
            scored = np.empty(queries.shape[0])
            for r in range(queries.shape[0]):
                t0 = time.perf_counter()
                a, s = index.query(queries[r:r + 1], items, allowed[r:r + 1], k, p)
                lat.append((time.perf_counter() - t0) * 1000)
                approx[r], scored[r] = a[0], s[0]
            rows.append(RetrievalRow(items.shape[0], "k-means index", c, p,
                                     float(recall_of_exact(approx, exact).mean()),
                                     float(scored.mean()), float(np.percentile(lat, 50))))
    return rows

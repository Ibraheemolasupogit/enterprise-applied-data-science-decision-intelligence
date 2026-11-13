"""Exact and approximate maximum-inner-product retrieval (Milestone 9).

* :func:`exact_top_k` — brute force: every allowed item's inner product, top ``k`` with ties to
  the lower item index (the M8 convention).
* :class:`KMeansIndex` — an inverted-file index: item embeddings are clustered by seeded
  k-means (Lloyd, fixed iterations); a query scores the centroids, searches the ``n_probe``
  clusters with the highest centroid inner product, and returns the exact top ``k`` among the
  allowed items in them. Clustering by Euclidean distance is a heuristic for inner-product
  search; the price of approximation is measured as recall of the exact top ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations.interactions import IntArray


def exact_top_k(query: FloatArray, items: FloatArray, allowed: np.ndarray, k: int) -> IntArray:
    """Exact top ``k`` items per query row (rows of ``allowed`` align with ``query``)."""
    scores = np.where(allowed, query @ items.T, -np.inf)
    if np.any(allowed.sum(axis=1) < k):
        raise ValueError("fewer than k allowed items")
    return np.asarray(np.argsort(-scores, axis=1, kind="stable")[:, :k], dtype=np.int64)


@dataclass
class KMeansIndex:
    n_clusters: int
    seed: int = 0
    iterations: int = 20
    centroids: FloatArray | None = None
    assignment: IntArray | None = None

    def build(self, items: FloatArray) -> KMeansIndex:
        if not 1 <= self.n_clusters <= items.shape[0]:
            raise ValueError("n_clusters must lie in [1, n_items]")
        rng = np.random.default_rng(self.seed)
        centroids = items[rng.choice(items.shape[0], self.n_clusters, replace=False)].copy()
        assign = np.zeros(items.shape[0], dtype=np.int64)
        for _ in range(self.iterations):
            d2 = ((items[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            assign = np.argmin(d2, axis=1)
            for c in range(self.n_clusters):
                members = items[assign == c]
                if members.size:
                    centroids[c] = members.mean(axis=0)
        self.centroids, self.assignment = centroids, assign
        return self

    def query(self, query: FloatArray, items: FloatArray, allowed: np.ndarray, k: int,
              n_probe: int) -> tuple[IntArray, np.ndarray]:
        """Top ``k`` per row among probed clusters; returns (items, n_scored per row).

        Rows with fewer than ``k`` allowed items in the probed clusters are padded with -1.
        """
        if self.centroids is None or self.assignment is None:
            raise RuntimeError("build the index first")
        n_probe = min(n_probe, self.n_clusters)
        probes = np.argsort(-(query @ self.centroids.T), axis=1, kind="stable")[:, :n_probe]
        out = np.full((query.shape[0], k), -1, dtype=np.int64)
        scored = np.zeros(query.shape[0], dtype=np.int64)
        for r in range(query.shape[0]):
            cand = np.flatnonzero(np.isin(self.assignment, probes[r]) & allowed[r])
            scored[r] = cand.size
            if cand.size == 0:
                continue
            s = items[cand] @ query[r]
            order = np.lexsort((cand, -s))[:k]
            out[r, :order.size] = cand[order]
        return out, scored


def recall_of_exact(approx: IntArray, exact: IntArray) -> FloatArray:
    """Share of each row's exact top-k found by the approximate search."""
    return np.asarray([np.isin(e, a[a >= 0]).mean() for a, e in zip(approx, exact, strict=True)],
                      dtype=np.float64)

"""Offline ranking, catalogue and exposure metrics (Brief §8.5) with explicit conventions.

Inputs: ``recs`` is a ``(n_users, K)`` integer array of item indices in rank order (rank 1
first), produced by a deterministic ranking with no ties left unresolved; ``relevant``
is a ``(n_users, n_items)`` boolean matrix and ``gains`` its graded (non-negative float)
counterpart. Every per-user metric returns one value per row; averaging, intervals and
comparisons happen in :mod:`.evaluation`.

Conventions (libraries differ on each; these are fixed here and in the methodology):

* **Precision@K** ``= |rec_K ∩ R| / K`` (denominator ``K`` even if fewer than ``K`` items
  are relevant).
* **Recall@K** ``= |rec_K ∩ R| / |R|``; undefined (NaN) for ``|R| = 0`` — callers drop
  such users.
* **HitRate@K** ``= 1{|rec_K ∩ R| > 0}``.
* **AP@K** ``= (1 / min(|R|, K)) sum_{k<=K} P@k * rel_k``; MAP@K is its mean over users.
  The ``min(|R|, K)`` denominator makes a perfect top-K list score 1.
* **RR@K** ``= 1 / rank`` of the first relevant item within the top K, 0 if none
  (MRR@K is the mean).
* **NDCG@K** ``= DCG@K / IDCG@K`` with ``DCG = sum_k g_k / log2(k + 1)`` (linear gain,
  not ``2^g - 1``) and ``IDCG`` the DCG of the best ``K`` gains among the items the
  model was allowed to rank; 0 when ``IDCG = 0``. Binary NDCG uses ``g = 1{relevant}``.
* **Catalogue coverage@K** = distinct recommended items / eligible items.
* **Exposure**: an item's exposure is the number of lists containing it, optionally
  position-weighted by ``1 / log2(k + 1)``; **Gini** (0 = equal, → 1 = concentrated),
  **normalised entropy** (1 = uniform) and the **top-share** (exposure share of the most
  exposed fraction of the eligible catalogue) are computed over *all* eligible items,
  zeros included.
* **Novelty** = mean self-information ``-log2((n_i + 1) / (n_users + 1))`` of recommended
  items, ``n_i`` = training users who clicked item ``i`` (add-one smoothed).
* **Intra-list diversity** = ``1 -`` mean pairwise cosine similarity of the content
  vectors in a list.
* **Personalisation** = ``1 -`` mean Jaccard overlap between the lists of all user pairs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decision_intelligence.common.validation import FloatArray

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _check(recs: IntArray, n_items: int) -> int:
    if recs.ndim != 2 or recs.shape[1] < 1:
        raise ValueError("recs must be a (n_users, K) array with K >= 1")
    if recs.size and (int(np.min(recs)) < 0 or int(np.max(recs)) >= n_items):
        raise ValueError("recommended item index out of range")
    if recs.shape[1] > 1 and np.any(np.diff(np.sort(recs, axis=1), axis=1) == 0):
        raise ValueError("a recommendation list repeats an item")
    return int(recs.shape[1])


def hits(recs: IntArray, relevant: BoolArray) -> BoolArray:
    """``rel_k`` for every rank: whether the item at rank ``k`` is relevant."""
    if relevant.shape[0] != recs.shape[0]:
        raise ValueError("recs and relevant must have the same number of rows")
    _check(recs, relevant.shape[1])
    return np.asarray(np.take_along_axis(relevant, recs, axis=1), dtype=np.bool_)


def precision_at_k(recs: IntArray, relevant: BoolArray) -> FloatArray:
    h = hits(recs, relevant)
    return np.asarray(h.sum(axis=1) / h.shape[1], dtype=np.float64)


def recall_at_k(recs: IntArray, relevant: BoolArray) -> FloatArray:
    h = hits(recs, relevant)
    n_rel = relevant.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(n_rel > 0, h.sum(axis=1) / n_rel, np.nan), dtype=np.float64)


def hit_rate_at_k(recs: IntArray, relevant: BoolArray) -> FloatArray:
    return np.asarray(hits(recs, relevant).any(axis=1), dtype=np.float64)


def average_precision_at_k(recs: IntArray, relevant: BoolArray) -> FloatArray:
    h = hits(recs, relevant).astype(np.float64)
    k = h.shape[1]
    precision_at_each = np.cumsum(h, axis=1) / np.arange(1, k + 1)
    denom = np.minimum(relevant.sum(axis=1), k).astype(np.float64)
    num = (precision_at_each * h).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(denom > 0, num / denom, np.nan), dtype=np.float64)


def reciprocal_rank_at_k(recs: IntArray, relevant: BoolArray) -> FloatArray:
    h = hits(recs, relevant)
    first = np.argmax(h, axis=1)
    return np.asarray(np.where(h.any(axis=1), 1.0 / (first + 1.0), 0.0), dtype=np.float64)


def discounts(k: int) -> FloatArray:
    return np.asarray(1.0 / np.log2(np.arange(2, k + 2)), dtype=np.float64)


def ndcg_at_k(recs: IntArray, gains: FloatArray, allowed: BoolArray | None = None) -> FloatArray:
    """NDCG@K with linear gains; the ideal ranking is over ``allowed`` items only."""
    if gains.shape[0] != recs.shape[0]:
        raise ValueError("recs and gains must have the same number of rows")
    k = _check(recs, gains.shape[1])
    if np.any(gains < 0) or not np.all(np.isfinite(gains)):
        raise ValueError("gains must be finite and non-negative")
    d = discounts(k)
    dcg = (np.take_along_axis(gains, recs, axis=1) * d).sum(axis=1)
    pool = gains if allowed is None else np.where(allowed, gains, 0.0)
    kk = min(k, pool.shape[1])
    ideal = -np.sort(-pool, axis=1)[:, :kk]
    idcg = (ideal * d[:kk]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(idcg > 0, dcg / idcg, 0.0), dtype=np.float64)


def catalogue_coverage(recs: IntArray, eligible: BoolArray) -> float:
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise ValueError("no eligible items")
    return float(np.unique(recs).size / n_eligible)


def exposure(recs: IntArray, n_items: int, *, position_weighted: bool = False) -> FloatArray:
    """Exposure per item (all ``n_items``; zeros for never-recommended items)."""
    k = _check(recs, n_items)
    w = discounts(k) if position_weighted else np.ones(k)
    return np.asarray(np.bincount(recs.ravel(), weights=np.tile(w, recs.shape[0]),
                                  minlength=n_items), dtype=np.float64)


def gini(values: FloatArray) -> float:
    """Gini coefficient of non-negative values (0 = perfectly equal)."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or np.any(x < 0):
        raise ValueError("gini needs non-negative values")
    total = x.sum()
    if total == 0:
        return 0.0
    n = x.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * x) / (n * total)) - (n + 1.0) / n)


def normalised_entropy(values: FloatArray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2 or x.sum() <= 0:
        raise ValueError("need at least two items and positive total exposure")
    p = x[x > 0] / x.sum()
    return float(-(p * np.log(p)).sum() / np.log(x.size))


def top_share(values: FloatArray, fraction: float = 0.1) -> float:
    """Share of total exposure taken by the top ``fraction`` of items (at least one)."""
    x = -np.sort(-np.asarray(values, dtype=np.float64))
    n_top = max(1, int(np.ceil(fraction * x.size)))
    return float(x[:n_top].sum() / x.sum()) if x.sum() > 0 else 0.0


def novelty(recs: IntArray, item_user_counts: FloatArray, n_users: int) -> FloatArray:
    _check(recs, item_user_counts.size)
    info = -np.log2((item_user_counts + 1.0) / (n_users + 1.0))
    return np.asarray(info[recs].mean(axis=1), dtype=np.float64)


def intra_list_diversity(recs: IntArray, item_features: FloatArray) -> FloatArray:
    """``1 -`` mean pairwise cosine over each list (exact, via ``|sum f|^2``)."""
    k = _check(recs, item_features.shape[0])
    if k < 2:
        raise ValueError("diversity needs K >= 2")
    norms = np.linalg.norm(item_features, axis=1, keepdims=True)
    f = np.divide(item_features, norms, out=np.zeros_like(item_features), where=norms > 0)
    g = f[recs]  # (n, K, F)
    total = np.einsum("nkf->nf", g)
    pair_sum = (np.einsum("nf,nf->n", total, total) - np.einsum("nkf,nkf->n", g, g)) / 2.0
    return np.asarray(1.0 - pair_sum / (k * (k - 1) / 2.0), dtype=np.float64)


def personalisation(recs: IntArray, n_items: int) -> float:
    """``1 -`` mean Jaccard overlap over all pairs of users' top-K lists."""
    k = _check(recs, n_items)
    n = recs.shape[0]
    if n < 2:
        raise ValueError("personalisation needs at least two users")
    b = np.zeros((n, n_items), dtype=np.float64)
    b[np.arange(n)[:, None], recs] = 1.0
    overlap = b @ b.T
    jaccard = overlap / (2.0 * k - overlap)
    upper = np.triu_indices(n, k=1)
    return float(1.0 - jaccard[upper].mean())


def rmse(predicted: FloatArray, actual: FloatArray) -> float:
    p, a = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if p.shape != a.shape or p.size == 0:
        raise ValueError("predicted and actual must be non-empty and the same shape")
    return float(np.sqrt(np.mean((p - a) ** 2)))

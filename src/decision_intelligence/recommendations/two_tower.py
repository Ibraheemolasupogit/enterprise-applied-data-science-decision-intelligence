"""A shallow two-tower retrieval model in NumPy (Milestone 9).

Two towers map *features* to a shared ``d``-dimensional space; the score is the inner product
``e_u . e_i``. Because both towers read features rather than ids, the model scores users with no
clicks and items launched after the cutoff (cold start) — its main reason to exist here.

* **User features** (observable at the cutoff): segment and region one-hot; the mean content
  vector of the user's clicked items; ``log(1 + clicks)``; a constant.
* **Item features**: the M8 content vector (tags + category, L2-normalised);
  ``log(1 + clicks before the cutoff)`` (0 for new items); a constant.
* **Tower**: linear (``e = W f``) or one hidden ReLU layer (``e = W2 relu(W1 f + b1)``).
* **Loss**: full softmax cross-entropy over all items launched by the cutoff, one term per
  (user, clicked item) pair, plus ``weight_decay/2 |W|^2``. A full softmax (not sampled
  negatives) avoids sampling bias at this catalogue size.
* **Optimiser**: Adam with seeded initialisation and mini-batches. Gradients are derived by
  hand and checked against finite differences in the tests.

This is a shallow model, not a production deep two-tower system; it is a NumPy implementation
so that no new dependency is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from decision_intelligence.common.validation import FloatArray
from decision_intelligence.recommendations.baselines import Recommender
from decision_intelligence.recommendations.interactions import IntArray, InteractionData


@dataclass
class Tower:
    """``e = W2 relu(W1 f + b1)`` (``hidden > 0``) or ``e = W f`` (``hidden == 0``)."""

    weights: list[FloatArray]

    @staticmethod
    def initialise(n_in: int, dim: int, hidden: int, rng: np.random.Generator) -> Tower:
        if hidden == 0:
            return Tower([rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_in, dim))])
        return Tower([rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, hidden)), np.zeros(hidden),
                      rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, dim))])

    def forward(self, f: FloatArray) -> tuple[FloatArray, tuple[FloatArray, ...]]:
        if len(self.weights) == 1:
            return f @ self.weights[0], (f,)
        pre = f @ self.weights[0] + self.weights[1]
        act = np.maximum(pre, 0.0)
        return act @ self.weights[2], (f, pre, act)

    def backward(self, cache: tuple[FloatArray, ...], d_out: FloatArray) -> list[FloatArray]:
        if len(self.weights) == 1:
            return [cache[0].T @ d_out]
        f, pre, act = cache
        d_act = d_out @ self.weights[2].T
        d_pre = d_act * (pre > 0)
        return [f.T @ d_pre, d_pre.sum(axis=0), act.T @ d_out]


def user_features(history: InteractionData, users: IntArray) -> FloatArray:
    segs = np.array(sorted(set(history.user_segment.tolist())))
    regs = np.array(sorted(set(history.user_region.tolist())))
    seg = (history.user_segment[users][:, None] == segs[None, :]).astype(np.float64)
    reg = (history.user_region[users][:, None] == regs[None, :]).astype(np.float64)
    clicks = history.clicks[users]
    n = clicks.sum(axis=1, keepdims=True)
    profile = np.divide(clicks @ history.item_features, n, out=np.zeros(
        (users.size, history.item_features.shape[1])), where=n > 0)
    return np.hstack([seg, reg, profile, np.log1p(n), np.ones((users.size, 1))])


def item_features(history: InteractionData) -> FloatArray:
    pop = np.log1p(history.clicks.sum(axis=0))[:, None]
    return np.hstack([history.item_features, pop, np.ones((history.n_items, 1))])


@dataclass
class TwoTowerParams:
    user: Tower
    item: Tower


def softmax_loss_and_grads(
    params: TwoTowerParams, fu: FloatArray, fi: FloatArray, positives: IntArray,
    allowed: np.ndarray, weight_decay: float,
) -> tuple[float, list[FloatArray], list[FloatArray]]:
    """Mean softmax cross-entropy of ``positives`` (item index per row of ``fu``) over the
    ``allowed`` items, plus weight decay; returns (loss, user-tower grads, item-tower grads)."""
    eu, cu = params.user.forward(fu)
    ei, ci = params.item.forward(fi)
    logits = eu @ ei.T
    logits = np.where(allowed[None, :], logits, -np.inf)
    m = logits.max(axis=1, keepdims=True)
    z = np.exp(logits - m)
    prob = z / z.sum(axis=1, keepdims=True)
    b = fu.shape[0]
    loss = float(-np.mean(np.log(prob[np.arange(b), positives])))
    loss += 0.5 * weight_decay * sum(float(np.sum(w ** 2)) for w in
                                     params.user.weights + params.item.weights)
    d_logits = prob.copy()
    d_logits[np.arange(b), positives] -= 1.0
    d_logits /= b
    gu = params.user.backward(cu, d_logits @ ei)
    gi = params.item.backward(ci, d_logits.T @ eu)
    gu = [g + weight_decay * w for g, w in zip(gu, params.user.weights, strict=True)]
    gi = [g + weight_decay * w for g, w in zip(gi, params.item.weights, strict=True)]
    return loss, gu, gi


@dataclass
class _Adam:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: list[FloatArray] = field(default_factory=list)
    v: list[FloatArray] = field(default_factory=list)

    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for k, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            mh = self.m[k] / (1 - self.beta1 ** self.t)
            vh = self.v[k] / (1 - self.beta2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class TwoTower(Recommender):
    def __init__(self, dim: int = 32, hidden: int = 0, *, epochs: int = 20,
                 learning_rate: float = 0.01, weight_decay: float = 1e-4, batch_size: int = 512,
                 seed: int = 0) -> None:
        super().__init__()
        if dim < 1 or hidden < 0 or epochs < 1 or learning_rate <= 0:
            raise ValueError("invalid two-tower hyperparameters")
        self.dim, self.hidden, self.epochs = dim, hidden, epochs
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.batch_size, self.seed = batch_size, seed
        self.name = f"two-tower (d={dim}, {'linear' if hidden == 0 else f'hidden {hidden}'})"
        self.params: TwoTowerParams | None = None
        self.loss_trace: list[float] = []
        self._item_embeddings: FloatArray = np.zeros((0, 0))

    def _fit(self, history: InteractionData) -> None:
        rng = np.random.default_rng(self.seed)
        all_users = np.arange(history.n_users)
        fu_all = user_features(history, all_users)
        fi = item_features(history)
        self.params = TwoTowerParams(Tower.initialise(fu_all.shape[1], self.dim, self.hidden, rng),
                                     Tower.initialise(fi.shape[1], self.dim, self.hidden, rng))
        u, i = np.nonzero(history.clicks > 0)
        allowed = history.item_launched
        opt_u, opt_i = _Adam(self.learning_rate), _Adam(self.learning_rate)
        self.loss_trace = []
        for _ in range(self.epochs):
            order = rng.permutation(u.size)
            total = 0.0
            for b in range(0, order.size, self.batch_size):
                sel = order[b:b + self.batch_size]
                loss, gu, gi = softmax_loss_and_grads(self.params, fu_all[u[sel]], fi, i[sel],
                                                      allowed, self.weight_decay)
                opt_u.step(self.params.user.weights, gu)
                opt_i.step(self.params.item.weights, gi)
                total += loss * sel.size
            self.loss_trace.append(total / max(u.size, 1))
        self._item_embeddings = self.params.item.forward(fi)[0]

    def user_embeddings(self, users: IntArray) -> FloatArray:
        assert self.params is not None
        return self.params.user.forward(user_features(self.history, users))[0]

    @property
    def item_embeddings(self) -> FloatArray:
        return self._item_embeddings

    def score(self, users: IntArray) -> FloatArray:
        return np.asarray(self.user_embeddings(users) @ self._item_embeddings.T, dtype=np.float64)

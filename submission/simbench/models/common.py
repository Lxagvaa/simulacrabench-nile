"""Shared helpers for the conditional models.

Every model implements fit(enc) -> self and predict(enc) -> {item: (n, k_cond)
probabilities}, i.e. the answer distribution given that the item was asked.
"""
import time

import numpy as np
from scipy.optimize import minimize

from ..data import Encoded

_DEADLINE = [float("inf")]

# Per-item cap for the expensive per-item models, set by the pipeline from
# the time budget (small in the development phase, all items in the final).
HEAVY_ITEM_CAP = [None]

_TORCH = []


def set_deadline(t: float):
    _DEADLINE[0] = t


def past_deadline() -> bool:
    return time.time() > _DEADLINE[0]


def torch_or_none():
    if not _TORCH:
        try:
            import torch
            _TORCH.append(torch)
        except Exception:
            _TORCH.append(None)
    return _TORCH[0]


class _Stop(Exception):
    pass


def lbfgs(fun, x0, maxiter: int, **options):
    """L-BFGS-B that respects the deadline and caps function evaluations.
    Returns (x, iterations); on timeout, the best point seen so far."""
    best = {"f": np.inf, "x": np.asarray(x0, dtype=float)}

    def wrapped(x, *a):
        if past_deadline():
            raise _Stop
        f, g = fun(x, *a)
        if np.isfinite(f) and f < best["f"]:
            best["f"], best["x"] = f, np.array(x, dtype=float)
        return f, g

    try:
        res = minimize(wrapped, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 3 * maxiter + 20, **options})
        return (res.x if np.isfinite(res.fun) else best["x"]), int(res.nit)
    except _Stop:
        return best["x"], -1


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def mi_keys(key: np.ndarray, y: np.ndarray, ky: int) -> float:
    """Miller-Madow corrected mutual information between integer keys and y."""
    n = len(y)
    if n < 10:
        return 0.0
    _, kc = np.unique(key, return_inverse=True)
    kx = int(kc.max()) + 1
    tab = np.bincount(kc * ky + y, minlength=kx * ky).reshape(kx, ky) / n
    px, py = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    nz = tab > 0
    mi = float((tab[nz] * np.log(tab[nz] / (px @ py)[nz])).sum())
    return mi - (nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2.0 * n)


def mutual_info(x: np.ndarray, kx: int, y: np.ndarray, ky: int) -> float:
    ok = (x >= 0) & (y >= 0)
    return mi_keys(x[ok], y[ok], ky)


def given_codes(enc: Encoded, dtype=np.float64, nan_missing: bool = True) -> np.ndarray:
    X = np.stack([enc.given_codes[c] for c in enc.given], 1).astype(dtype)
    if nan_missing:
        X[X < 0] = np.nan
    return X


class Marginal:
    """Smoothed answer shares among respondents who were asked."""
    name = "marginal"

    def __init__(self, pseudo: float = 0.5):
        self.pseudo = pseudo

    def fit(self, enc: Encoded):
        self.p = {}
        for j in enc.predict:
            y = enc.y_cond(j)
            k = enc.items[j].k_cond
            cnt = np.bincount(y[y >= 0], minlength=k).astype(float) + self.pseudo
            self.p[j] = cnt / cnt.sum()
        return self

    def predict(self, enc: Encoded):
        return {j: np.tile(p, (enc.n, 1)) for j, p in self.p.items()}


def descendants(enc: Encoded) -> dict:
    """Number of PREDICT items gated, directly or through a chain, on each item."""
    kids: dict[str, list] = {}
    for j in enc.predict:
        p = enc.items[j].parent
        if p is not None and p in enc.items and enc.items[p].cls == "PREDICT":
            kids.setdefault(p, []).append(j)
    memo: dict[str, int] = {}

    def count(j, seen=()):
        if j not in memo:
            memo[j] = sum(1 + count(c, seen + (j,)) for c in kids.get(j, []) if c not in seen)
        return memo[j]
    return {j: count(j) for j in enc.predict}


def informative_items(enc: Encoded, min_rows: int, max_classes: int | None = None) -> list:
    """[(score, item)] for items worth a per-item model; score = MI of the top
    three GIVEN columns times the number of labelled rows."""
    items, cand = enc.items, []
    for j in enc.predict:
        y, k = enc.y_cond(j), items[j].k_cond
        n_lab = int((y >= 0).sum())
        if k <= 1 or (max_classes and k > max_classes) or n_lab < min_rows \
                or len(np.unique(y[y >= 0])) < 2:
            continue
        info = sum(sorted((mutual_info(enc.given_codes[c], items[c].k_full, y, k)
                           for c in enc.given), reverse=True)[:3])
        cand.append((info * n_lab, j))
    return cand


def rank_items(enc: Encoded, cand: list, max_items: int) -> list:
    """Gate parents first (by number of dependent items), then by score."""
    lev = descendants(enc)
    order = sorted(cand, key=lambda t: (lev.get(t[1], 0), t[0]), reverse=True)
    return [j for _, j in order[:max_items]]

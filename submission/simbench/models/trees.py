"""Tree ensembles: per-item gradient boosting and a multi-output random forest."""
import os

import numpy as np

from ..data import Encoded
from .common import Marginal, given_codes, informative_items, past_deadline, rank_items

GBDT_PARAMS = dict(learning_rate=0.15, max_iter=150, max_leaf_nodes=15, min_samples_leaf=20,
                   l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                   n_iter_no_change=10, random_state=0)


def _n_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _fit_hgb(X, y, cat, params):
    # single-threaded; items are fitted in parallel processes instead
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        m = HistGradientBoostingClassifier(categorical_features=cat, **params)
        m.fit(X, y)
    return m


class GBDT:
    """HistGradientBoosting per item on the GIVEN codes (native categoricals).
    Items that are not fitted fall back to the marginal."""

    def __init__(self, max_items: int = 1000, min_rows: int = 150, eps: float = 0.02,
                 params: dict | None = None, name: str | None = None):
        self.max_items, self.min_rows, self.eps = max_items, min_rows, eps
        self.params = {**GBDT_PARAMS, **(params or {})}
        self.name = name or "gbdt"

    def fit(self, enc: Encoded):
        items = enc.items
        self.base = Marginal().fit(enc).p
        chosen = rank_items(enc, informative_items(enc, self.min_rows), self.max_items)
        X = given_codes(enc)
        cat = [items[c].k_full <= 255 for c in enc.given]
        tasks = []
        for j in chosen:
            y = enc.y_cond(j)
            # early stopping needs at least two rows per class for its stratified split
            ok = y >= 0
            ok &= np.bincount(y[ok], minlength=items[j].k_cond)[np.clip(y, 0, None)] >= 2
            if len(np.unique(y[ok])) >= 2:
                tasks.append((j, ok))
        self.models = {}
        n_jobs = _n_jobs()
        if n_jobs > 1 and len(tasks) > n_jobs:
            from joblib import Parallel, delayed
            with Parallel(n_jobs=n_jobs, backend="loky") as par:
                for s0 in range(0, len(tasks), 4 * n_jobs):
                    if past_deadline():
                        break
                    batch = tasks[s0:s0 + 4 * n_jobs]
                    fitted = par(delayed(_fit_hgb)(X[ok], enc.y_cond(j)[ok], cat, self.params)
                                 for j, ok in batch)
                    self.models.update({j: m for (j, _), m in zip(batch, fitted)})
        else:
            for j, ok in tasks:
                if past_deadline():
                    break
                self.models[j] = _fit_hgb(X[ok], enc.y_cond(j)[ok], cat, self.params)
        return self

    def predict(self, enc: Encoded):
        X = given_codes(enc) if self.models else None
        out = {}
        for j, base in self.base.items():
            if j in self.models:
                m = self.models[j]
                p = np.zeros((enc.n, len(base)))
                p[:, m.classes_] = m.predict_proba(X)
                out[j] = (1.0 - self.eps) * p + self.eps * base
            else:
                out[j] = np.tile(base, (enc.n, 1))
        return out


class RandomForest:
    """One multi-output random forest for all items. Every leaf stores a
    distribution for every item, so tree count and leaf size are kept modest
    (memory grows with both)."""

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 5,
                 max_features: float = 0.5, max_samples: float | None = 0.5,
                 eps: float = 0.03, name: str | None = None):
        self.n_estimators, self.min_samples_leaf = n_estimators, min_samples_leaf
        self.max_features, self.max_samples, self.eps = max_features, max_samples, eps
        self.name = name or "rf"

    def fit(self, enc: Encoded):
        from sklearn.ensemble import RandomForestClassifier
        self.base = Marginal().fit(enc).p
        self.items = [j for j in enc.predict if enc.items[j].k_cond > 1]
        if not self.items:
            self.model = None
            return self
        Y = np.stack([enc.y_cond(j) for j in self.items], 1)
        Y = np.where(Y < 0, -1, Y)                       # not asked is its own class
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, max_samples=self.max_samples,
            n_jobs=_n_jobs(), random_state=0)
        self.model.fit(given_codes(enc, np.float32, nan_missing=False), Y)
        return self

    def predict(self, enc: Encoded):
        out = {j: np.tile(b, (enc.n, 1)) for j, b in self.base.items()}
        if self.model is None:
            return out
        probs = self.model.predict_proba(given_codes(enc, np.float32, nan_missing=False))
        for j, P, cls in zip(self.items, probs, self.model.classes_):
            base = self.base[j]
            p = np.zeros((enc.n, len(base)))
            keep = cls >= 0
            p[:, cls[keep].astype(int)] = P[:, keep]
            s = p.sum(1, keepdims=True)
            p = np.where(s > 0, p / np.clip(s, 1e-12, None), base)
            out[j] = (1.0 - self.eps) * p + self.eps * base
        return out

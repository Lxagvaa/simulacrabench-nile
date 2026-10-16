"""TabPFN v2 (Prior Labs) per item, from the bundled checkpoint."""
import os

import numpy as np

from ..data import Encoded
from .common import (HEAVY_ITEM_CAP, Marginal, given_codes, informative_items, past_deadline,
                     rank_items)

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "weights", "tabpfn-v2-classifier.ckpt")


class TabPFN:
    """In-context classifier per item. Training rows are capped at `max_rows`
    (the v2 context size); items with more than `max_classes` options use the
    marginal. A single classifier is re-fitted per item at predict time, which
    keeps memory at one copy of the network. Needs a GPU."""

    def __init__(self, max_items: int = 40, max_rows: int = 10000, max_classes: int = 10,
                 n_estimators: int = 4, min_rows: int = 150, eps: float = 0.02,
                 use_cap: bool = True, name: str | None = None):
        self.max_items, self.max_rows, self.max_classes = max_items, max_rows, max_classes
        self.n_estimators, self.min_rows, self.eps = n_estimators, min_rows, eps
        self.use_cap = use_cap
        self.name = name or "tabpfn"

    @staticmethod
    def _device():
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("TabPFN needs a GPU")

    def fit(self, enc: Encoded):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from tabpfn import TabPFNClassifier
        if not os.path.exists(CHECKPOINT):
            raise FileNotFoundError(CHECKPOINT)
        device = self._device()
        self.base = Marginal().fit(enc).p
        cap = HEAVY_ITEM_CAP[0] if (self.use_cap and HEAVY_ITEM_CAP[0] is not None) \
            else self.max_items
        chosen = rank_items(enc, informative_items(enc, self.min_rows, self.max_classes), cap)
        X = given_codes(enc)
        self.clf = TabPFNClassifier(model_path=CHECKPOINT, device=device,
                                    n_estimators=self.n_estimators,
                                    categorical_features_indices=list(range(X.shape[1])),
                                    ignore_pretraining_limits=True, random_state=0)
        rng = np.random.default_rng(0)
        self.context = {}
        for j in chosen:
            y = enc.y_cond(j)
            rows = np.nonzero(y >= 0)[0]
            if len(rows) > self.max_rows:
                rows = np.sort(rng.choice(rows, self.max_rows, replace=False))
            self.context[j] = (X[rows], y[rows])
        if self.context:                                  # fail early if the network won't load
            self.clf.fit(*next(iter(self.context.values())))
        return self

    def predict(self, enc: Encoded):
        X = given_codes(enc) if self.context else None
        out = {}
        for j, base in self.base.items():
            if j in self.context and not past_deadline():
                self.clf.fit(*self.context[j])
                p = np.zeros((enc.n, len(base)))
                p[:, np.asarray(self.clf.classes_, dtype=int)] = self.clf.predict_proba(X)
                out[j] = (1.0 - self.eps) * p + self.eps * base
            else:
                out[j] = np.tile(base, (enc.n, 1))
        return out

"""Linear pool of the components with EM-fitted weights, plus temperature calibration.

Weights are fitted per item on the out-of-fold log score and shrunk toward one
global weight vector per instrument (Dirichlet prior of strength tau).
"""
import numpy as np
from scipy.optimize import minimize_scalar

FLOOR = 1e-3
T_GRID = np.round(np.arange(0.6, 1.501, 0.05), 3)


def floor_mix(p: np.ndarray) -> np.ndarray:
    """The grader's renormalisation and flat floor."""
    p = p / p.sum(axis=1, keepdims=True)
    return FLOOR + (1.0 - p.shape[1] * FLOOR) * p


def _em(L: np.ndarray, prior: np.ndarray, tau: float, iters: int = 300,
        tol: float = 1e-6, init: np.ndarray | None = None) -> np.ndarray:
    """MAP-EM for mixture weights; L is (members, rows), the likelihood of the
    true answer under each member."""
    n = L.shape[1]
    w = (prior if init is None else init).copy()
    for _ in range(iters):
        w_new = (w * (L / (w @ L)).sum(1) + tau * prior) / (n + tau)
        if np.abs(w_new - w).max() < tol:
            return w_new
        w = w_new
    return w


class Blender:
    def __init__(self, names: list, tau: float = 30.0):
        self.names, self.tau = names, tau

    def fit(self, oof: dict, y_full: dict):
        M = len(self.names)
        per_item = {}
        for j, y in y_full.items():
            ok = y >= 0
            if ok.any():
                per_item[j] = np.stack([floor_mix(oof[m][j][ok])[np.arange(ok.sum()), y[ok]]
                                        for m in self.names])
        allL = np.concatenate(list(per_item.values()), axis=1)
        pick = np.random.default_rng(0).choice(allL.shape[1], min(allL.shape[1], 200_000),
                                               replace=False)
        sub = allL[:, pick]
        # start next to the best single member: EM is slow along flat directions
        init = np.full(M, 0.2 / M)
        init[int(np.log(sub).sum(1).argmax())] += 0.8
        self.global_w = _em(sub, np.full(M, 1.0 / M), tau=0.0, iters=2000, tol=1e-8, init=init)
        self.w = {j: _em(L, self.global_w, self.tau) for j, L in per_item.items()}

        self.report = {m: float(np.log(allL[i]).mean()) for i, m in enumerate(self.names)}
        blend = np.concatenate([(self.w[j][:, None] * L).sum(0) for j, L in per_item.items()])
        self.report["BLEND"] = float(np.log(blend).mean())
        self._fit_temperature(oof, y_full)
        return self

    def _fit_temperature(self, oof: dict, y_full: dict):
        """One temperature per instrument in [0.6, 1.5]; an item gets its own
        only if that gains more than 2 nats out of fold."""
        logs, ys, js = [], [], []
        for j, y in y_full.items():
            ok = y >= 0
            if not ok.any():
                continue
            p = self.combine({m: {j: oof[m][j][ok]} for m in self.names}, j)
            logs.append(np.log(np.clip(p / p.sum(1, keepdims=True), 1e-12, None)))
            ys.append(y[ok])
            js.append(j)

        def item_ll(lp, y, T):
            q = floor_mix(np.exp(lp / T))
            return float(np.log(q[np.arange(len(y)), y]).sum())

        def neg(T):
            return -sum(item_ll(lp, y, T) for lp, y in zip(logs, ys)) / max(1, sum(map(len, ys)))

        self.T, self.T_item = 1.0, {}
        if not logs:
            return
        res = minimize_scalar(neg, bounds=(0.6, 1.5), method="bounded", options={"xatol": 1e-3})
        if res.fun < neg(1.0):
            self.T = float(res.x)
        for lp, y, j in zip(logs, ys, js):
            base = item_ll(lp, y, self.T)
            scores = [item_ll(lp, y, T) for T in T_GRID]
            k = int(np.argmax(scores))
            if scores[k] - base > 2.0:
                self.T_item[j] = float(T_GRID[k])

    def calibrate(self, p: np.ndarray, item: str | None = None) -> np.ndarray:
        T = self.T_item.get(item, self.T)
        if T == 1.0:
            return p
        lp = np.log(np.clip(p / p.sum(1, keepdims=True), 1e-12, None)) / T
        lp -= lp.max(1, keepdims=True)
        q = np.exp(lp)
        return q / q.sum(1, keepdims=True)

    def combine(self, preds: dict, item: str) -> np.ndarray:
        w = self.w.get(item, self.global_w)
        return sum(w[i] * preds[m][item] for i, m in enumerate(self.names))

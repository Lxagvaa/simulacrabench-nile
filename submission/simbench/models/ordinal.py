"""Proportional-odds model for ordered answer scales."""
import re

import numpy as np

from ..data import Encoded
from .common import Marginal, lbfgs, past_deadline

NONSUBSTANTIVE = re.compile(
    r"don.?t know|refus|prefer not|no answer|not answered|no response|not recorded|not sure|"
    r"not applicable|^n/?a$|^9[6-9]\.|^-?9{2,}", re.I)


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class OrdinalLogit:
    """P(y <= k | x) = sigmoid(theta_k - x . beta) over an item's substantive
    options in schema order; "don't know"-type options keep their marginal
    share. Items with fewer than `min_levels` substantive options use the
    marginal."""
    name = "ordinal"

    def __init__(self, l2: float = 1e-2, maxiter: int = 100, min_levels: int = 3):
        self.l2, self.maxiter, self.min_levels = l2, maxiter, min_levels

    def _fit_item(self, X, y, K):
        n, D = X.shape
        cnt = np.bincount(y, minlength=K) + 0.5
        cum = np.cumsum(cnt)[:-1] / cnt.sum()
        th0 = np.log(cum / (1 - cum))
        # thresholds parameterised as t0, then log-increments (keeps them ordered)
        t0 = np.concatenate([[th0[0]], np.log(np.maximum(np.diff(th0), 1e-3))])
        x0 = np.concatenate([np.zeros(D), t0])
        lo, hi = y - 1, y

        def f(par):
            beta, t = par[:D], par[D:]
            th = np.cumsum(np.concatenate([[t[0]], np.exp(t[1:])]))
            eta = X @ beta
            thp = np.concatenate([[-np.inf], th, [np.inf]])
            a, b = thp[hi + 1] - eta, thp[lo + 1] - eta
            Fa, Fb = _sigmoid(a), _sigmoid(b)
            p = np.clip(Fa - Fb, 1e-12, None)
            fa = np.where(np.isfinite(a), Fa * (1 - Fa), 0.0)
            fb = np.where(np.isfinite(b), Fb * (1 - Fb), 0.0)
            loss = -np.log(p).sum() / n + 0.5 * self.l2 * (beta @ beta)
            g_beta = X.T @ ((fa - fb) / p / n) + self.l2 * beta
            g_th = np.zeros(K - 1)
            np.add.at(g_th, np.clip(hi, 0, K - 2), np.where(hi <= K - 2, -fa / p, 0.0) / n)
            np.add.at(g_th, np.clip(lo, 0, K - 2), np.where(lo >= 0, fb / p, 0.0) / n)
            rev = np.cumsum(g_th[::-1])[::-1]
            g_t = np.empty(K - 1)
            g_t[0] = rev[0]
            g_t[1:] = rev[1:] * np.exp(t[1:])
            return loss, np.concatenate([g_beta, g_t])

        x, _ = lbfgs(f, x0, self.maxiter)
        beta, t = x[:D], x[D:]
        return beta, np.cumsum(np.concatenate([[t[0]], np.exp(t[1:])]))

    def fit(self, enc: Encoded):
        X = enc.X.astype(np.float64)
        self.base = Marginal().fit(enc).p
        self.params = {}
        for j in enc.predict:
            it = enc.items[j]
            vals = list(it.values)[: it.k_cond]
            subst = np.array([not NONSUBSTANTIVE.search(str(v)) for v in vals])
            K = int(subst.sum())
            y = enc.y_cond(j)
            if past_deadline():
                break
            if K < self.min_levels or (y >= 0).sum() < 50:
                continue
            level = np.cumsum(subst) - 1
            ok = (y >= 0) & subst[np.clip(y, 0, None)]
            if len(np.unique(y[ok])) < 2:
                continue
            self.params[j] = (subst, *self._fit_item(X[ok], level[y[ok]], K))
        return self

    def predict(self, enc: Encoded):
        X = enc.X.astype(np.float64)
        out = {}
        for j, base in self.base.items():
            if j not in self.params:
                out[j] = np.tile(base, (enc.n, 1))
                continue
            subst, beta, th = self.params[j]
            F = _sigmoid(th[None, :] - (X @ beta)[:, None])
            F = np.concatenate([np.zeros((enc.n, 1)), F, np.ones((enc.n, 1))], axis=1)
            p = np.tile(base, (enc.n, 1))
            p[:, subst] = base[subst].sum() * np.diff(F, axis=1)
            out[j] = p
        return out

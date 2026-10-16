"""Count-based models: hierarchical backoff and profile matching."""
import re

import numpy as np

from ..data import Encoded
from .common import Marginal, mi_keys

ALPHA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


class Backoff:
    """Hierarchical Dirichlet smoothing over the most informative GIVEN columns:

        p_0 = marginal
        p_l = (counts in stratum(c_1..c_l) + alpha * p_{l-1}) / (N + alpha)

    Columns are added greedily by conditional mutual information; alpha is
    chosen per item by leave-one-out log score.
    """

    def __init__(self, depth: int = 2, alpha: float | None = None, name: str | None = None):
        self.depth, self.alpha = depth, alpha
        self.name = name or f"backoff_d{depth}"

    def _choose_cols(self, enc: Encoded, y: np.ndarray, k: int) -> list:
        items, cols = enc.items, []
        valid = y >= 0
        key = np.zeros(enc.n, dtype=np.int64)
        base_mi = 0.0
        for _ in range(self.depth):
            best, best_gain = None, 0.0
            for c in enc.given:
                if c in cols:
                    continue
                codes = enc.given_codes[c]
                ok = valid & (codes >= 0)
                if ok.sum() < 10:
                    continue
                gain = mi_keys(key[ok] * items[c].k_full + codes[ok], y[ok], k) - base_mi
                if gain > best_gain:
                    best, best_gain = c, gain
            if best is None:
                break
            cols.append(best)
            codes = enc.given_codes[best]
            valid = valid & (codes >= 0)
            key = key * items[best].k_full + np.clip(codes, 0, None)
            base_mi += best_gain
        return cols

    @staticmethod
    def _loo_alpha(cnt0: np.ndarray, pseudo: float, y: np.ndarray, levels: list) -> float:
        k, n0 = len(cnt0), cnt0.sum()
        p_root = (cnt0[y] - 1.0 + pseudo) / (n0 - 1.0 + k * pseudo)
        best, best_ll = 4.0, -np.inf
        for a in ALPHA_GRID:
            p = p_root.copy()
            for ok, c_true, n_strat in levels:
                p = np.where(ok, (c_true - 1.0 + a * p) / (n_strat - 1.0 + a), p)
            ll = float(np.log(np.clip(p, 1e-12, None)).mean())
            if ll > best_ll:
                best, best_ll = a, ll
        return best

    def fit(self, enc: Encoded):
        items, pseudo = enc.items, 0.5
        self.base = Marginal(pseudo).fit(enc).p
        self.plan = {}
        for j in enc.predict:
            y, k = enc.y_cond(j), items[j].k_cond
            if k <= 1 or (y >= 0).sum() < 2:
                self.plan[j] = ([], [], 4.0)
                continue
            cols = self._choose_cols(enc, y, k)
            tables, levels, radix = [], [], 1
            key = np.zeros(enc.n, dtype=np.int64)
            valid = y >= 0
            lab = np.nonzero(valid)[0]
            for c in cols:
                kc = items[c].k_full
                codes = enc.given_codes[c]
                valid = valid & (codes >= 0)
                key = key * kc + np.clip(codes, 0, None)
                radix *= kc
                cnt = np.bincount(key[valid] * k + y[valid], minlength=radix * k)
                cnt = cnt.reshape(radix, k).astype(float)
                tables.append((radix, cnt))
                kl, yl = key[lab], y[lab]
                levels.append((valid[lab], cnt[kl, yl], cnt[kl].sum(1)))
            if self.alpha is not None or not cols:
                alpha = self.alpha or 4.0
            else:
                cnt0 = np.bincount(y[lab], minlength=k).astype(float)
                alpha = self._loo_alpha(cnt0, pseudo, y[lab], levels)
            self.plan[j] = (cols, tables, alpha)
        return self

    def predict(self, enc: Encoded):
        items, out = enc.items, {}
        for j, (cols, tables, alpha) in self.plan.items():
            p = np.tile(self.base[j], (enc.n, 1))
            key = np.zeros(enc.n, dtype=np.int64)
            ok = np.ones(enc.n, dtype=bool)
            for c, (radix, cnt) in zip(cols, tables):
                codes = enc.given_codes[c]
                ok = ok & (codes >= 0)
                key = key * items[c].k_full + np.clip(codes, 0, None)
                counts = cnt[np.clip(key, 0, radix - 1)]
                counts[~ok] = 0.0
                p = (counts + alpha * p) / (counts.sum(1, keepdims=True) + alpha)
            out[j] = p
        return out


class ProfileMatch:
    """Answers of respondents with the same GIVEN profile:

        p1 = (counts among rows differing in at most one column + a1 * base) / (N1 + a1)
        p  = (counts among rows with the identical profile    + a2 * p1)   / (N2 + a2)

    base is a depth-3 Backoff; (a1, a2) per item by leave-one-out. The survey
    round column is never the one allowed to differ, so a household is not
    matched to its own record from the other round.
    """

    ROUND_COLUMN = re.compile(r"^(wave|round)$|_wave$|_round$", re.I)

    def __init__(self, base_depth: int = 3, name: str | None = None):
        self.base_depth = base_depth
        self.name = name or "profile"

    @staticmethod
    def _profiles(enc: Encoded) -> np.ndarray:
        if not enc.given:
            return np.zeros((enc.n, 1), np.int64)
        return np.stack([enc.given_codes[c] for c in enc.given], 1)

    @staticmethod
    def _index(G: np.ndarray):
        uniq, inv = np.unique(G, axis=0, return_inverse=True)
        return {tuple(r): i for i, r in enumerate(uniq)}, inv.ravel(), len(uniq)

    def fit(self, enc: Encoded):
        self.base_model = Backoff(depth=self.base_depth).fit(enc)
        base = self.base_model.predict(enc)
        G = self._profiles(enc)
        self.full = self._index(G)
        self.cols = [i for i, c in enumerate(enc.given) if not self.ROUND_COLUMN.search(c)]
        ncol = len(self.cols)
        self.drop = [self._index(np.delete(G, c, axis=1)) for c in self.cols]
        self.tab_full, self.tab_drop, self.alpha = {}, {}, {}
        for j in enc.predict:
            y, k = enc.y_cond(j), enc.items[j].k_cond
            lab = y >= 0
            if k <= 1 or lab.sum() < 2:
                continue
            yl = y[lab]

            def counts(inv, nk):
                return np.bincount(inv[lab] * k + yl, minlength=nk * k).reshape(nk, k).astype(float)

            cf = counts(self.full[1], self.full[2])
            cds = [counts(inv, nk) for _, inv, nk in self.drop]
            self.tab_full[j], self.tab_drop[j] = cf, cds
            # leave-one-out counts for every labelled row
            r = np.arange(lab.sum())
            f_rows = cf[self.full[1][lab]]
            f_rows[r, yl] -= 1.0
            d_rows = sum(cd[inv[lab]] for cd, (_, inv, _) in zip(cds, self.drop))
            d_rows[r, yl] -= ncol
            d_rows -= (ncol - 1) * f_rows                 # exact matches appear in every key
            d_rows = np.clip(d_rows, 0.0, None)
            b = base[j][lab]
            best, best_ll = (4.0, 4.0), -np.inf
            for a1 in ALPHA_GRID:
                p1 = (d_rows + a1 * b) / (d_rows.sum(1, keepdims=True) + a1)
                for a2 in ALPHA_GRID:
                    p = (f_rows[r, yl] + a2 * p1[r, yl]) / (f_rows.sum(1) + a2)
                    ll = float(np.log(np.clip(p, 1e-12, None)).mean())
                    if ll > best_ll:
                        best, best_ll = (a1, a2), ll
            self.alpha[j] = best
        return self

    def predict(self, enc: Encoded):
        base = self.base_model.predict(enc)
        G = self._profiles(enc)
        fi = np.array([self.full[0].get(tuple(r), -1) for r in G])
        ncol = len(self.cols)
        di = [np.array([lk.get(tuple(r), -1) for r in np.delete(G, c, axis=1)])
              for c, (lk, _, _) in zip(self.cols, self.drop)]
        out = {}
        for j, p in base.items():
            if j not in self.tab_full:
                out[j] = p
                continue
            f = np.zeros_like(p)
            f[fi >= 0] = self.tab_full[j][fi[fi >= 0]]
            d = np.zeros_like(p)
            for idx, cd in zip(di, self.tab_drop[j]):
                d[idx >= 0] += cd[idx[idx >= 0]]
            d = np.clip(d - (ncol - 1) * f, 0.0, None)
            a1, a2 = self.alpha[j]
            p1 = (d + a1 * p) / (d.sum(1, keepdims=True) + a1)
            out[j] = (f + a2 * p1) / (f.sum(1, keepdims=True) + a2)
        return out

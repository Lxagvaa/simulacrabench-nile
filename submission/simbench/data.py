"""Frame encoding and gate-aware composition.

For a gated item, P(y = v) = P(asked) * P(v | asked) and P(y = sentinel) =
1 - P(asked). The conditional models only see rows where the item was asked;
GateModel estimates P(asked) from the parent's predicted distribution.
Items are composed in topological order so gate chains propagate.
"""
import numpy as np
import pandas as pd


def _key(v):
    # the same option may arrive as 3, 3.0 or "3"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    return str(v)


def codes_for(column: pd.Series, options: list) -> np.ndarray:
    """Integer codes over `options`; NaN and unknown values become -1."""
    lookup = {_key(v): i for i, v in enumerate(options)}
    vals = column.to_numpy(dtype=object)
    miss = pd.isna(vals)
    return np.fromiter((-1 if m else lookup.get(_key(v), -1) for v, m in zip(vals, miss)),
                       dtype=np.int64, count=len(column))


class Encoded:
    """Integer-coded view of the frame.

    given_codes[c]  codes of GIVEN column c
    y_full[j]       codes of PREDICT item j over values + sentinel, -1 if unknown
    blank[j]        True where the cell is NaN, i.e. has to be predicted
    X               one-hot design matrix of the GIVEN block
    """

    def __init__(self, frame: pd.DataFrame, items: dict, given: list, predict: list):
        self.n = len(frame)
        self.given, self.predict, self.items = given, predict, items
        self.given_codes = {c: codes_for(frame[c], items[c].options) for c in given}
        self.y_full = {j: codes_for(frame[j], items[j].options) for j in predict}
        # an off-schema visible value is -1 in y_full but must not count as blank
        self.blank = {j: frame[j].isna().to_numpy() for j in predict}

        blocks = []
        for c in given:
            block = np.zeros((self.n, items[c].k_full), dtype=np.float32)
            codes = self.given_codes[c]
            ok = codes >= 0
            block[np.nonzero(ok)[0], codes[ok]] = 1.0
            blocks.append(block)
        self.X = np.concatenate(blocks, axis=1) if blocks else np.zeros((self.n, 0), np.float32)

    def y_cond(self, j: str) -> np.ndarray:
        """Codes given that the item was asked; -1 where skipped or unknown."""
        y = self.y_full[j].copy()
        y[y >= self.items[j].k_cond] = -1
        return y

    def subset(self, rows: np.ndarray) -> "Encoded":
        out = Encoded.__new__(Encoded)
        out.n = len(rows)
        out.given, out.predict, out.items = self.given, self.predict, self.items
        out.given_codes = {c: v[rows] for c, v in self.given_codes.items()}
        out.y_full = {j: v[rows] for j, v in self.y_full.items()}
        out.blank = {j: v[rows] for j, v in self.blank.items()}
        out.X = self.X[rows]
        return out


def _mi_bin(key: np.ndarray, k: int, asked: np.ndarray) -> float:
    """Miller-Madow corrected mutual information between codes in [0, k) and a 0/1 flag."""
    n = len(asked)
    if n < 10:
        return 0.0
    tab = np.bincount(key * 2 + asked, minlength=2 * k).reshape(k, 2) / n
    px, py = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    nz = tab > 0
    mi = float((tab[nz] * np.log(tab[nz] / (px @ py)[nz])).sum())
    return mi - (nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2.0 * n)


class GateModel:
    """P(asked | parent value, up to two GIVEN context columns), estimated from
    the visible rows with the schema rule as a prior:

        T0[v]          = (asked | parent=v + s * rule[v]) / (n + s)
        T1[v, c1]      = (asked | v, c1 + m * T0[v]) / (n + m)
        T2[v, c1, c2]  = (asked | v, c1, c2 + m * T1[v, c1]) / (n + m)

    Real files do not always follow the routing (e.g. one survey round writes
    "0" where the other writes the sentinel), hence the empirical tables and
    the context columns, picked greedily by conditional mutual information.
    """

    def __init__(self, prior_strength: float = 2.0, rule_conf: float = 0.995,
                 ctx_strength: float = 10.0, max_ctx: int = 2):
        self.prior_strength = prior_strength
        self.rule_conf = rule_conf
        self.ctx_strength = ctx_strength
        self.max_ctx = max_ctx
        self.spec: dict[str, tuple] = {}

    @staticmethod
    def _best_ctx(enc, key, size, valid, asked, exclude):
        if valid.sum() < 30:
            return None
        base = _mi_bin(key[valid], size, asked[valid])
        best_c, best_gain = None, 1e-4
        for c in enc.given:
            if c in exclude:
                continue
            cc, kc = enc.given_codes[c], enc.items[c].k_full
            mm = valid & (cc >= 0)
            gain = _mi_bin(key[mm] * kc + cc[mm], size * kc, asked[mm]) - base
            if gain > best_gain:
                best_c, best_gain = c, gain
        return best_c

    def fit(self, enc) -> "GateModel":
        items = enc.items
        for j in enc.predict:
            it = items[j]
            if not it.gated:
                continue
            y = enc.y_full[j]
            seen = y >= 0
            asked = ((y < it.k_cond) & seen).astype(np.int64)
            p = it.parent if it.parent in items else None
            if p is None:
                pc, kp = np.zeros(enc.n, dtype=np.int64), 1
                prior = np.full(1, (asked[seen].sum() + 1.0) / (seen.sum() + 2.0))
            else:
                pc = enc.given_codes[p] if items[p].cls == "GIVEN" else enc.y_full[p]
                kp = items[p].k_full
                prior = np.array([self.rule_conf if v in it.observed_if else 1 - self.rule_conf
                                  for v in items[p].options])
            s, m = self.prior_strength, self.ctx_strength
            ok = seen & (pc >= 0)
            n_v = np.bincount(pc[ok], minlength=kp).astype(float)
            a_v = np.bincount(pc[ok], weights=asked[ok], minlength=kp)
            T0 = (a_v + s * prior) / (n_v + s)

            ctxs, tables = [], []
            key, size, prev, valid = pc.copy(), kp, T0, ok.copy()
            for _ in range(self.max_ctx):
                c = self._best_ctx(enc, key, size, valid, asked, exclude={p, *ctxs})
                if c is None:
                    break
                cc, kc = enc.given_codes[c], items[c].k_full
                valid = valid & (cc >= 0)
                key = key * kc + np.clip(cc, 0, None)
                size *= kc
                n_k = np.bincount(key[valid], minlength=size).astype(float)
                a_k = np.bincount(key[valid], weights=asked[valid], minlength=size)
                T = (a_k + m * np.repeat(prev, kc)) / (n_k + m)
                ctxs.append(c)
                tables.append(T)
                prev = T
            self.spec[j] = (p, kp, T0, ctxs, tables)
        return self

    def p_asked(self, j: str, enc, full: dict) -> np.ndarray:
        items = enc.items
        p, kp, T0, ctxs, tables = self.spec[j]
        # rows[i, v] = P(asked | parent = v, context of row i)
        rows = np.tile(T0, (enc.n, 1))
        key = np.tile(np.arange(kp), (enc.n, 1))
        ok = np.ones(enc.n, dtype=bool)
        for c, T in zip(ctxs, tables):
            cc, kc = enc.given_codes[c], items[c].k_full
            ok = ok & (cc >= 0)
            key = key * kc + np.clip(cc, 0, None)[:, None]
            rows = np.where(ok[:, None], T[key], rows)
        if p is None:
            return rows[:, 0]
        if items[p].cls == "GIVEN":
            pc = enc.given_codes[p]
            pa = rows[np.arange(enc.n), np.clip(pc, 0, kp - 1)]
            return np.where(pc >= 0, pa, rows.mean(1))
        return (full[p] * rows).sum(1)

    def compose(self, cond: dict, enc, order: list) -> dict:
        """cond[j] over the answers -> full[j] over answers + sentinel."""
        items, full = enc.items, {}
        for j in order:
            it, c = items[j], cond[j]
            if not it.gated:
                full[j] = c
                continue
            pa = np.clip(self.p_asked(j, enc, full), 0.0, 1.0)[:, None]
            full[j] = np.concatenate([pa * c, 1.0 - pa], axis=1)
        return full


def normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, None)
    return p / p.sum(axis=1, keepdims=True)

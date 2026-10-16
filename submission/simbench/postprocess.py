"""Structure-aware corrections of the blended predictions.

Each step is fitted on the out-of-fold blend and applied to an item or group
only if it improves the out-of-fold log score by more than `min_gain` nats.
"""
import re

import numpy as np

from .blend import floor_mix
from .models import NONSUBSTANTIVE

T_GRID = np.round(np.arange(0.6, 1.501, 0.05), 3)


def _ll(P: np.ndarray, y: np.ndarray) -> float:
    ok = y >= 0
    if not ok.any():
        return 0.0
    q = floor_mix(P[ok])
    return float(np.log(q[np.arange(ok.sum()), y[ok]]).sum())


def _temper(P: np.ndarray, T: float) -> np.ndarray:
    lp = np.log(np.clip(P, 1e-12, None)) / T
    lp -= lp.max(1, keepdims=True)
    Q = np.exp(lp)
    return Q / Q.sum(1, keepdims=True)


# ---------------------------------------------------------------- gate parents

def _rescale_child(C: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Scale the asked mass of a child distribution (sentinel last)."""
    asked = 1.0 - C[:, -1]
    new = np.clip(asked * ratio, 0.0, 1.0 - 1e-6)
    cond = C[:, :-1] / np.clip(asked, 1e-12, None)[:, None]
    out = np.empty_like(C)
    out[:, :-1] = cond * new[:, None]
    out[:, -1] = 1.0 - new
    return out


class SubtreeCalibrator:
    """Temperature per gate parent, chosen on the parent's log score plus its
    children's: a parent's distribution also sets each child's chance of
    being asked, via t(v) = P(child asked | parent = v)."""

    def __init__(self, min_gain: float = 5.0, min_children: int = 2):
        self.min_gain, self.min_children = min_gain, min_children
        self.params: dict[str, tuple] = {}

    def fit(self, items: dict, predict: list, oof: dict, y_full: dict):
        kids: dict[str, list] = {}
        for j in predict:
            p = items[j].parent
            if items[j].gated and p in items and items[p].cls == "PREDICT" and j in oof:
                kids.setdefault(p, []).append(j)
        for p, ch in kids.items():
            if len(ch) < self.min_children or p not in oof:
                continue
            yp = y_full[p]
            kp = oof[p].shape[1]
            tables = {}
            for c in ch:
                yc = y_full[c]
                m = (yp >= 0) & (yc >= 0)
                asked = (yc[m] < items[c].k_cond).astype(float)
                tables[c] = (np.bincount(yp[m], weights=asked, minlength=kp) + 0.5) \
                    / (np.bincount(yp[m], minlength=kp) + 1.0)
            base_den = {c: oof[p] @ tables[c] for c in ch}

            def total(T):
                if T == 1.0:
                    return _ll(oof[p], yp) + sum(_ll(oof[c], y_full[c]) for c in ch)
                Pp = _temper(oof[p], T)
                tot = _ll(Pp, yp)
                for c in ch:
                    ratio = (Pp @ tables[c]) / np.clip(base_den[c], 1e-12, None)
                    tot += _ll(_rescale_child(oof[c], ratio), y_full[c])
                return tot

            base = total(1.0)
            best_T, best = 1.0, base
            for T in T_GRID:
                v = total(float(T))
                if v > best:
                    best_T, best = float(T), v
            if best - base > self.min_gain:
                self.params[p] = (best_T, ch, tables)
        return self

    def apply(self, full: dict) -> dict:
        out = dict(full)
        for p, (T, ch, tables) in self.params.items():
            Pp = _temper(out[p], T)
            for c in ch:
                ratio = (Pp @ tables[c]) / np.clip(out[p] @ tables[c], 1e-12, None)
                out[c] = _rescale_child(out[c], ratio)
            out[p] = Pp
        return out


# ------------------------------------------------------- multi-select families

def find_families(schema: dict, predict: list, min_size: int = 3) -> list:
    """0/1 items that share a question stem before "[option:" ("tick all that apply")."""
    groups: dict[str, list] = {}
    for j in predict:
        rec = schema["items"][j]
        q = rec.get("question") or ""
        if "[option:" in q and [str(v) for v in rec["values"]] == ["0", "1"]:
            groups.setdefault(q.split("[option:")[0].strip(), []).append(j)
    return [ms for ms in groups.values() if len(ms) >= min_size]


def _split(P: np.ndarray):
    asked = P[:, 0] + P[:, 1]
    q = P[:, 1] / np.clip(asked, 1e-12, None)
    return asked, np.clip(q, 1e-6, 1 - 1e-6)


def _shift(Q: np.ndarray, target: np.ndarray, iters: int = 40) -> np.ndarray:
    """Common logit shift per row so that the tick probabilities sum to target."""
    L = np.log(Q / (1 - Q))
    lo, hi = np.full(len(Q), -12.0), np.full(len(Q), 12.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = (1.0 / (1.0 + np.exp(-(L + mid[:, None])))).sum(1)
        hi = np.where(s > target, mid, hi)
        lo = np.where(s > target, lo, mid)
    return 1.0 / (1.0 + np.exp(-(L + 0.5 * (lo + hi)[:, None])))


def _adjust_family(full: dict, members: list, a: float, b: float) -> dict:
    asked, Q = zip(*(_split(full[j]) for j in members))
    Q = np.stack(Q, 1)
    target = np.clip(a + b * Q.sum(1), 0.05, len(members) - 0.05)
    Qn = _shift(Q, target)
    out = {}
    for i, j in enumerate(members):
        P = full[j].copy()
        P[:, 1] = asked[i] * Qn[:, i]
        P[:, 0] = asked[i] * (1 - Qn[:, i])
        out[j] = P
    return out


class FamilyAdjust:
    """Ties each respondent's expected number of ticks in a family to the
    observed relation true_count ~ a + b * predicted_count."""

    def __init__(self, min_gain: float = 5.0):
        self.min_gain = min_gain
        self.params: dict[tuple, tuple] = {}

    def fit(self, families: list, oof: dict, y_full: dict):
        for members in families:
            members = [j for j in members if oof[j].shape[1] >= 2]
            if len(members) < 3:
                continue
            Y = np.stack([y_full[j] for j in members], 1)
            rows = np.all((Y == 0) | (Y == 1), axis=1)
            if rows.sum() < 50:
                continue
            Q = np.stack([_split(oof[j][rows])[1] for j in members], 1)
            s_pred, s_true = Q.sum(1), (Y[rows] == 1).sum(1).astype(float)
            var = s_pred.var()
            b = float(np.clip(np.cov(s_pred, s_true)[0, 1] / var, 0.0, 1.2)) if var > 1e-9 else 0.0
            a = float(s_true.mean() - b * s_pred.mean())
            adjusted = _adjust_family(oof, members, a, b)
            gain = sum(_ll(adjusted[j], y_full[j]) - _ll(oof[j], y_full[j]) for j in members)
            if gain > self.min_gain:
                self.params[tuple(members)] = (a, b)
        return self

    def apply(self, full: dict) -> dict:
        out = dict(full)
        for members, (a, b) in self.params.items():
            out.update(_adjust_family(out, list(members), a, b))
        return out


# ------------------------------------------------------------- ordinal scales

NUMERIC = re.compile(r"^\s*(<|>|≤|≥)?\s*-?\d+(\.\d+)?\s*(\+|-\s*-?\d+(\.\d+)?)?\s*$")
SMOOTH_GRID = [(w, r) for w in (0.05, 0.1, 0.2, 0.3) for r in (1, 2, 4)]


def find_ordinal(schema: dict, predict: list, min_levels: int = 3) -> dict:
    """Items whose substantive options are numeric bands -> their column indices."""
    out = {}
    for j in predict:
        vals = [str(v) for v in schema["items"][j]["values"]]
        idx = [i for i, v in enumerate(vals) if not NONSUBSTANTIVE.search(v)]
        if len(idx) >= min_levels and all(NUMERIC.match(vals[i]) for i in idx):
            out[j] = np.array(idx)
    return out


def _smooth(P: np.ndarray, idx: np.ndarray, w: float, r: int) -> np.ndarray:
    s = P[:, idx].copy()
    for _ in range(r):
        left = np.concatenate([s[:, :1], s[:, :-1]], axis=1)
        right = np.concatenate([s[:, 1:], s[:, -1:]], axis=1)
        s = (1 - 2 * w) * s + w * left + w * right
    out = P.copy()
    out[:, idx] = s
    return out


class OrdinalSmoother:
    """Diffuses predicted mass between neighbouring bands of a binned scale."""

    def __init__(self, min_gain: float = 2.0):
        self.min_gain = min_gain
        self.params: dict[str, tuple] = {}

    def fit(self, ordinal: dict, oof: dict, y_full: dict):
        for j, idx in ordinal.items():
            if j not in oof:
                continue
            base = _ll(oof[j], y_full[j])
            best, best_ll = None, base
            for w, r in SMOOTH_GRID:
                v = _ll(_smooth(oof[j], idx, w, r), y_full[j])
                if v > best_ll:
                    best, best_ll = (w, r), v
            if best is not None and best_ll - base > self.min_gain:
                self.params[j] = (idx, *best)
        return self

    def apply(self, full: dict) -> dict:
        out = dict(full)
        for j, (idx, w, r) in self.params.items():
            out[j] = _smooth(out[j], idx, w, r)
        return out

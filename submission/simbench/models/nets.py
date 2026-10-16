"""Multi-task softmax regression / MLP over all items at once."""
import copy

import numpy as np
from scipy import sparse

from ..data import Encoded
from .common import lbfgs, mutual_info, torch_or_none


def pair_spec(enc: Encoded, top: int = 5, min_count: int = 20) -> list:
    """One-hot cells for every pair of the `top` most informative GIVEN columns,
    keeping cells with at least `min_count` rows."""
    items = enc.items
    score = []
    for c in enc.given:
        kc = items[c].k_full
        tot = sum(mutual_info(enc.given_codes[c], kc, enc.y_cond(j), items[j].k_cond)
                  for j in enc.predict if items[j].k_cond > 1)
        score.append((tot, c))
    cols = [c for _, c in sorted(score, reverse=True)[:top]]
    spec, off = [], 0
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            c1, c2 = cols[a], cols[b]
            k2 = items[c2].k_full
            x1, x2 = enc.given_codes[c1], enc.given_codes[c2]
            ok = (x1 >= 0) & (x2 >= 0)
            cnt = np.bincount(x1[ok] * k2 + x2[ok], minlength=items[c1].k_full * k2)
            lookup = np.full(len(cnt), -1, dtype=np.int64)
            keep = np.nonzero(cnt >= min_count)[0]
            lookup[keep] = off + np.arange(len(keep))
            off += len(keep)
            spec.append((c1, c2, k2, lookup))
    return spec


def pair_X(enc: Encoded, spec: list) -> np.ndarray:
    F = sum(int((lk >= 0).sum()) for *_, lk in spec)
    out = np.zeros((enc.n, F), dtype=np.float32)
    for c1, c2, k2, lookup in spec:
        x1, x2 = enc.given_codes[c1], enc.given_codes[c2]
        ok = (x1 >= 0) & (x2 >= 0)
        idx = np.full(enc.n, -1, dtype=np.int64)
        idx[ok] = lookup[x1[ok] * k2 + x2[ok]]
        r = np.nonzero(idx >= 0)[0]
        out[r, idx[r]] = 1.0
    return out


def _with_pairs(enc: Encoded, spec: list) -> Encoded:
    out = copy.copy(enc)
    out.X = np.concatenate([enc.X, pair_X(enc, spec)], axis=1)
    return out


class MultiTaskNet:
    """Shared (optional tanh) hidden layer, one softmax head per item, loss
    masked to rows where the item was asked. hidden=0 is multinomial logistic
    regression. Full-batch L-BFGS; output biases start at the log marginals.
    Uses torch on a GPU when available, numpy otherwise.
    """

    def __init__(self, hidden: int = 0, l2: float = 3e-3, maxiter: int = 250,
                 seed: int = 0, name: str | None = None, backend: str = "auto",
                 pairs: int = 0):
        self.hidden, self.l2, self.maxiter, self.seed = hidden, l2, maxiter, seed
        self.backend, self.pairs = backend, pairs
        base = f"mlp{hidden}_l2{l2:g}" if hidden else f"linear_l2{l2:g}"
        if pairs:
            base += f"_pairs{pairs}"
        elif seed and hidden:
            base += f"_s{seed}"
        self.name = name or base

    def _layout(self, enc: Encoded):
        # items grouped by option count, so each group's softmax is one reshape
        self.items = [j for j in enc.predict if enc.items[j].k_cond > 1]
        self.trivial = [j for j in enc.predict if enc.items[j].k_cond <= 1]
        groups: dict[int, list] = {}
        for j in self.items:
            groups.setdefault(enc.items[j].k_cond, []).append(j)
        self.groups, off = [], 0
        for k, js in sorted(groups.items()):
            self.groups.append((k, js, off))
            off += k * len(js)
        self.C = off

    def _prepare(self, enc: Encoded):
        n = enc.n
        self.Xs = sparse.csr_matrix(enc.X.astype(np.float32))
        mask = np.zeros((n, self.C), dtype=np.float32)
        tgt = []
        for k, js, off in self.groups:
            for t, j in enumerate(js):
                y = enc.y_cond(j)
                ok = y >= 0
                c0 = off + t * k
                mask[ok, c0:c0 + k] = 1.0
                tgt.append(np.nonzero(ok)[0] * self.C + c0 + y[ok])
        self.Mcol = mask
        self.tgt = np.concatenate(tgt) if tgt else np.zeros(0, np.int64)
        self.Mgrp = [mask[:, off:off + k * len(js)].reshape(n, len(js), k)[:, :, 0].copy()
                     for k, js, off in self.groups]

    def _unpack(self, theta):
        D, h, C = self.D, self.hidden, self.C
        i = 0
        if h:
            W1 = theta[i:i + D * h].reshape(D, h); i += D * h
            b1 = theta[i:i + h]; i += h
            W2 = theta[i:i + h * C].reshape(h, C); i += h * C
        else:
            W1 = b1 = None
            W2 = theta[i:i + D * C].reshape(D, C); i += D * C
        return W1, b1, W2, theta[i:i + C]

    def _probs(self, Z):
        n = Z.shape[0]
        E = np.exp(np.clip(Z, -30.0, 30.0))
        P = np.empty_like(E)
        logS = []
        for k, js, off in self.groups:
            Eg = E[:, off:off + k * len(js)].reshape(n, len(js), k)
            S = Eg[:, :, 0].copy()
            for c in range(1, k):
                S += Eg[:, :, c]
            P[:, off:off + k * len(js)] = (Eg / S[:, :, None]).reshape(n, -1)
            logS.append(np.log(S))
        return P, logS

    def _loss_grad(self, theta, n_norm):
        W1, b1, W2, b2 = self._unpack(theta.astype(np.float32))
        if self.hidden:
            A = np.tanh(np.asarray(self.Xs @ W1) + b1)
            Z = A @ W2 + b2
        else:
            A = None
            Z = np.asarray(self.Xs @ W2) + b2
        P, logS = self._probs(Z)
        loss = sum(float((ls * m).sum()) for ls, m in zip(logS, self.Mgrp))
        loss -= float(np.clip(Z, -30.0, 30.0).ravel()[self.tgt].sum())
        dZ = P * self.Mcol
        dZ.ravel()[self.tgt] -= 1.0
        loss /= n_norm
        dZ /= n_norm
        gb2 = dZ.sum(0)
        if self.hidden:
            gW2 = A.T @ dZ + self.l2 * W2
            dA = (dZ @ W2.T) * (1.0 - A * A)
            gW1 = np.asarray(self.Xs.T @ dA) + self.l2 * W1
            loss += 0.5 * self.l2 * (float((W1 * W1).sum()) + float((W2 * W2).sum()))
            grad = np.concatenate([gW1.ravel(), dA.sum(0), gW2.ravel(), gb2])
        else:
            gW2 = np.asarray(self.Xs.T @ dZ) + self.l2 * W2
            loss += 0.5 * self.l2 * float((W2 * W2).sum())
            grad = np.concatenate([gW2.ravel(), gb2])
        return loss, grad.astype(np.float64)

    def _log_marginals(self, enc: Encoded) -> np.ndarray:
        b2 = np.zeros(self.C)
        for k, js, off in self.groups:
            for t, j in enumerate(js):
                y = enc.y_cond(j)
                cnt = np.bincount(y[y >= 0], minlength=k) + 0.5
                b2[off + t * k: off + (t + 1) * k] = np.log(cnt / cnt.sum())
        return b2

    def fit(self, enc: Encoded):
        self.spec = pair_spec(enc, self.pairs) if self.pairs else None
        if self.spec:
            enc = _with_pairs(enc, self.spec)
        torch = torch_or_none()
        if self.backend == "torch" or (self.backend == "auto" and torch is not None
                                       and torch.cuda.is_available()):
            return self._fit_torch(enc)
        self._layout(enc)
        self._prepare(enc)
        self.D = enc.X.shape[1]
        b2 = self._log_marginals(enc)
        if self.hidden:
            rng = np.random.default_rng(self.seed)
            scale = 1.0 / np.sqrt(max(1.0, float(enc.X.sum(1).mean())))
            W1 = rng.normal(0, scale, (self.D, self.hidden))
            theta0 = np.concatenate([W1.ravel(), np.zeros(self.hidden),
                                     np.zeros(self.hidden * self.C), b2])
        else:
            theta0 = np.concatenate([np.zeros(self.D * self.C), b2])
        x, _ = lbfgs(lambda th: self._loss_grad(th, float(enc.n)), theta0, self.maxiter, maxcor=20)
        self.theta = x.astype(np.float32)
        del self.Xs, self.Mcol, self.tgt, self.Mgrp
        return self

    def _fit_torch(self, enc: Encoded):
        torch = torch_or_none()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        self._layout(enc)
        n, D, h, C = enc.n, enc.X.shape[1], self.hidden, self.C
        self.D = D
        X = torch.as_tensor(enc.X, dtype=torch.float32, device=dev)
        Ys = [torch.as_tensor(np.stack([enc.y_cond(j) for j in js], 1), device=dev)
              for k, js, off in self.groups]
        b2 = torch.tensor(self._log_marginals(enc).astype(np.float32), device=dev,
                          requires_grad=True)
        if h:
            scale = 1.0 / np.sqrt(max(1.0, float(enc.X.sum(1).mean())))
            W1 = (torch.randn(D, h, device=dev) * scale).requires_grad_()
            b1 = torch.zeros(h, device=dev, requires_grad=True)
            W2 = torch.zeros(h, C, device=dev, requires_grad=True)
            params, reg = [W1, b1, W2, b2], [W1, W2]
        else:
            W2 = torch.zeros(D, C, device=dev, requires_grad=True)
            params, reg = [W2, b2], [W2]

        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=self.maxiter, history_size=20,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-10)

        def closure():
            opt.zero_grad()
            A = torch.tanh(X @ W1 + b1) if h else X
            Z = A @ W2 + b2
            loss = 0.0
            for (k, js, off), Y in zip(self.groups, Ys):
                lp = torch.log_softmax(Z[:, off:off + k * len(js)].view(n, len(js), k), dim=2)
                ll = lp.gather(2, Y.clamp(min=0).unsqueeze(2)).squeeze(2)
                loss = loss - (ll * (Y >= 0)).sum()
            loss = loss / n + 0.5 * self.l2 * sum((w * w).sum() for w in reg)
            loss.backward()
            return loss

        opt.step(closure)
        flat = [W1, b1, W2, b2] if h else [W2, b2]
        self.theta = np.concatenate([p.detach().cpu().numpy().ravel() for p in flat]).astype(np.float32)
        return self

    def predict(self, enc: Encoded):
        if self.spec:
            enc = _with_pairs(enc, self.spec)
        W1, b1, W2, b2 = self._unpack(self.theta)
        X = enc.X.astype(np.float32)
        Z = (np.tanh(X @ W1 + b1) @ W2 + b2) if self.hidden else (X @ W2 + b2)
        P, _ = self._probs(Z)
        out, n = {}, enc.n
        for k, js, off in self.groups:
            Pg = P[:, off:off + k * len(js)].reshape(n, len(js), k)
            for t, j in enumerate(js):
                out[j] = Pg[:, t, :].astype(np.float64)
        for j in self.trivial:
            out[j] = np.ones((n, max(1, enc.items[j].k_cond)))
        return out

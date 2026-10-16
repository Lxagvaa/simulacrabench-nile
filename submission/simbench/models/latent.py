"""Latent-structure models fitted on all answers jointly; prediction uses GIVEN only."""
import numpy as np
from scipy import sparse

from ..data import Encoded
from .common import lbfgs, past_deadline, softmax, torch_or_none


class LatentClass:
    """Latent class regression with K respondent types:

        P(type | GIVEN) = softmax(X W + b),   P(y_j | type c) = theta_j[c]

    fitted by EM over the one-hot answer matrix.
    """

    def __init__(self, K: int = 8, iters: int = 80, l2: float = 1e-2, smooth: float = 2.0,
                 seed: int = 0, name: str | None = None):
        self.K, self.iters, self.l2, self.smooth, self.seed = K, iters, l2, smooth, seed
        self.name = name or f"latent_k{K}"

    def _fit_prior(self, X, R, W, b):
        n, D = X.shape
        K = self.K

        def f(theta):
            W_, b_ = theta[:D * K].reshape(D, K), theta[D * K:]
            Z = X @ W_ + b_
            Zm = Z.max(1, keepdims=True)
            lse = np.log(np.exp(Z - Zm).sum(1, keepdims=True)) + Zm
            loss = -(R * (Z - lse)).sum() / n + 0.5 * self.l2 * (W_ * W_).sum()
            G = (np.exp(Z - lse) - R) / n
            return loss, np.concatenate([(X.T @ G + self.l2 * W_).ravel(), G.sum(0)])

        x, _ = lbfgs(f, np.concatenate([W.ravel(), b]), 15)
        return x[:D * K].reshape(D, K), x[D * K:]

    def fit(self, enc: Encoded):
        items = enc.items
        n, K = enc.n, self.K
        self.items = [j for j in enc.predict if items[j].k_cond > 1]
        self.trivial = [j for j in enc.predict if items[j].k_cond <= 1]
        self.offsets, off = {}, 0
        rows, cols, marg = [], [], []
        for j in self.items:
            k = items[j].k_cond
            y = enc.y_cond(j)
            ok = np.nonzero(y >= 0)[0]
            rows.append(ok)
            cols.append(off + y[ok])
            cnt = np.bincount(y[ok], minlength=k) + 0.5
            marg.append(cnt / cnt.sum())
            self.offsets[j] = (off, k)
            off += k
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        Y1 = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, off))
        prior = self.smooth * np.concatenate(marg) if marg else np.zeros(0)
        X = enc.X.astype(np.float64)
        D = X.shape[1]

        R = np.random.default_rng(self.seed).dirichlet(np.full(K, 0.3), size=n)
        W, b = np.zeros((D, K)), np.zeros(K)
        prev = -np.inf
        for it in range(self.iters):
            if it and past_deadline():
                break
            C = np.asarray((Y1.T @ R).T) + prior
            logT = np.empty_like(C)
            for j in self.items:
                o, k = self.offsets[j]
                blk = C[:, o:o + k]
                logT[:, o:o + k] = np.log(blk / blk.sum(1, keepdims=True))
            W, b = self._fit_prior(X, R, W, b)
            Z = X @ W + b
            logpi = Z - (np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1, keepdims=True))
                         + Z.max(1, keepdims=True))
            L = logpi + np.asarray(Y1 @ logT.T)
            Lm = L.max(1, keepdims=True)
            ll = float((np.log(np.exp(L - Lm).sum(1)) + Lm[:, 0]).mean())
            R = softmax(L)
            if ll - prev < 1e-5 * abs(ll):
                break
            prev = ll
        self.W, self.b = W, b
        self.theta = {j: np.exp(logT[:, o:o + k]) for j, (o, k) in self.offsets.items()}
        return self

    def predict(self, enc: Encoded):
        pi = softmax(enc.X.astype(np.float64) @ self.W + self.b)
        out = {j: pi @ T for j, T in self.theta.items()}
        for j in self.trivial:
            out[j] = np.ones((enc.n, max(1, enc.items[j].k_cond)))
        return out


class FactorVAE:
    """Continuous latent traits (multidimensional nominal-response IRT):

        z ~ N(G x, I),   P(y_j = k | z) = softmax_k(a_jk + B_jk . z)

    trained as a variational autoencoder whose encoder sees the answers;
    predictions average the decoder over the GIVEN-dependent prior.
    """

    def __init__(self, d: int = 3, hidden: int = 128, epochs: int = 400, lr: float = 3e-2,
                 l2: float = 1e-4, samples: int = 64, seed: int = 0, name: str | None = None):
        self.d, self.hidden, self.epochs, self.lr = d, hidden, epochs, lr
        self.l2, self.samples, self.seed = l2, samples, seed
        self.name = name or f"factor_d{d}"

    def fit(self, enc: Encoded):
        torch = torch_or_none()
        if torch is None:
            raise RuntimeError("FactorVAE needs torch")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        items = enc.items
        self.items = [j for j in enc.predict if items[j].k_cond > 1]
        self.trivial = [j for j in enc.predict if items[j].k_cond <= 1]
        self.offsets, off = {}, 0
        for j in self.items:
            self.offsets[j] = (off, items[j].k_cond)
            off += items[j].k_cond
        C, n, D, d = off, enc.n, enc.X.shape[1], self.d
        Yoh = np.zeros((n, C), np.float32)
        tgt = np.full((n, len(self.items)), -1, np.int64)
        a0 = np.zeros(C, np.float32)
        for t, j in enumerate(self.items):
            o, k = self.offsets[j]
            y = enc.y_cond(j)
            ok = y >= 0
            Yoh[np.nonzero(ok)[0], o + y[ok]] = 1.0
            tgt[:, t] = np.where(ok, o + y, -1)
            cnt = np.bincount(y[ok], minlength=k) + 0.5
            a0[o:o + k] = np.log(cnt / cnt.sum())
        T = torch.as_tensor
        X = T(enc.X.astype(np.float32), device=dev)
        Yo = T(Yoh, device=dev)
        tg = T(tgt, device=dev)
        mask = tg >= 0
        seg = T(np.concatenate([[t] * self.offsets[j][1] for t, j in enumerate(self.items)]),
                device=dev)
        nn = torch.nn
        self.G = nn.Linear(D, d).to(dev)
        self.encoder = nn.Sequential(nn.Linear(D + C, self.hidden), nn.Tanh(),
                                     nn.Linear(self.hidden, 2 * d)).to(dev)
        self.A = nn.Parameter(T(a0, device=dev))
        self.B = nn.Parameter(0.01 * torch.randn(d, C, device=dev))
        params = list(self.G.parameters()) + list(self.encoder.parameters()) + [self.A, self.B]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.l2)
        nitems = len(self.items)

        def seg_logsoftmax(Z):
            Zmax = torch.zeros(Z.shape[0], nitems, device=dev).scatter_reduce(
                1, seg.expand(Z.shape[0], -1), Z, reduce="amax", include_self=False)
            E = torch.exp(Z - Zmax[:, seg])
            S = torch.zeros(Z.shape[0], nitems, device=dev).index_add_(1, seg, E)
            return Z - Zmax[:, seg] - torch.log(S)[:, seg]

        for ep in range(self.epochs):
            if ep and ep % 25 == 0 and past_deadline():
                break
            opt.zero_grad()
            h = self.encoder(torch.cat([X, Yo], 1))
            mu, logv = h[:, :d], h[:, d:].clamp(-6, 3)
            z = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
            logp = seg_logsoftmax(self.A + z @ self.B)
            ll = torch.where(mask, logp.gather(1, tg.clamp(min=0)),
                             torch.zeros_like(tg, dtype=logp.dtype))
            kl = 0.5 * (torch.exp(logv) + (mu - self.G(X)) ** 2 - 1.0 - logv).sum(1)
            loss = (-ll.sum(1) + kl).mean()
            loss.backward()
            opt.step()
        self._seg_logsoftmax = seg_logsoftmax
        self.dev = dev
        return self

    def predict(self, enc: Encoded):
        torch = torch_or_none()
        with torch.no_grad():
            X = torch.as_tensor(enc.X.astype(np.float32), device=self.dev)
            pm = self.G(X)
            gen = torch.Generator(device=self.dev).manual_seed(self.seed)
            acc = None
            for _ in range(self.samples):
                z = pm + torch.randn(pm.shape, generator=gen, device=self.dev)
                P = torch.exp(self._seg_logsoftmax(self.A + z @ self.B))
                acc = P if acc is None else acc + P
            P = (acc / self.samples).cpu().numpy().astype(np.float64)
        out = {j: P[:, o:o + k] for j, (o, k) in self.offsets.items()}
        for j in self.trivial:
            out[j] = np.ones((enc.n, max(1, enc.items[j].k_cond)))
        return out

"""Head-to-head holdout comparison of single components on a sandbox instrument.

    .venv-tabpfn/bin/python scripts/compare_models.py unhcr_panel gbdt rf tabpfn1

Fits each model on 70% of the visible rows and reports the mean log score of
the true answer (conditional on asked, floored like the grader) on the other
30%, plus fit+predict time. Single models only: the blend is not involved.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "submission"))
import local_eval as le  # noqa: E402
from simbench import models as M  # noqa: E402
from simbench.data import Encoded  # noqa: E402
from simbench.schema_utils import parse_schema  # noqa: E402

inst, names = sys.argv[1], sys.argv[2:]
frame, schema, truth, scored = le.build(os.path.join(ROOT, "starter_kit"), inst, 1)
items, given, predict = parse_schema(schema)
enc = Encoded(frame, items, given, predict)
B = np.stack([enc.blank[j] for j in predict], 1).any(1)
vis = enc.subset(np.nonzero(~B)[0])
perm = np.random.default_rng(0).permutation(vis.n)
tr, va = vis.subset(perm[: int(0.7 * vis.n)]), vis.subset(perm[int(0.7 * vis.n):])
M.HEAVY_ITEM_CAP[0] = 1000

FACTORIES = {
    "mlp": lambda: M.MultiTaskNet(hidden=64, l2=3e-3),
    "gbdt": lambda: M.GBDT(),
    "gbdt_deep": lambda: M.GBDT(params=dict(max_leaf_nodes=31, min_samples_leaf=10)),
    "rf": lambda: M.RandomForest(),
    "latent8": lambda: M.LatentClass(K=8),
    "latent16": lambda: M.LatentClass(K=16, seed=1),
    "factor2": lambda: M.FactorVAE(d=2),
    "factor3": lambda: M.FactorVAE(d=3),
    "factor5": lambda: M.FactorVAE(d=5),
    "tabpfn1": lambda: M.TabPFN(max_items=1000, n_estimators=1, use_cap=False),
    "tabpfn4": lambda: M.TabPFN(max_items=1000, n_estimators=4, use_cap=False),
}
for name in names:
    t = time.time()
    P = FACTORIES[name]().fit(tr).predict(va)
    lls = []
    for j in predict:
        y = va.y_cond(j)
        ok = y >= 0
        if items[j].k_cond > 1 and ok.any() and j in P:
            p = P[j][ok] / P[j][ok].sum(1, keepdims=True)
            p = 1e-3 + (1 - p.shape[1] * 1e-3) * p
            lls.append(np.log(p[np.arange(ok.sum()), y[ok]]).mean())
    print(f"{inst:12s} {name:10s} mean log score {np.mean(lls):.4f}  ({time.time() - t:6.1f}s)",
          flush=True)

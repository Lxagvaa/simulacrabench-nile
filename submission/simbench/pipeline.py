"""frame + schema -> one probability vector per blank cell.

1. encode the frame; visible rows are the complete ones
2. K-fold out-of-fold predictions for every component that fits in the budget
3. blend weights and temperatures on the OOF predictions
4. refit on all visible rows (or average the fold models if time is short)
5. structure-aware post-processing: gate parents, multi-select families, ordinal scales
6. vectors in canonical order (rows top to bottom, items in schema order)
"""
import time

import numpy as np
import pandas as pd

from .blend import Blender
from .data import Encoded, GateModel, normalize
from .models import (GBDT, HEAVY_ITEM_CAP, Backoff, Direct, FactorVAE, LatentClass, Marginal,
                     MultiTaskNet, OrdinalLogit, ProfileMatch, RandomForest, TabPFN, set_deadline)
from .postprocess import (FamilyAdjust, OrdinalSmoother, SubtreeCalibrator, find_families,
                          find_ordinal)
from .schema_utils import parse_schema, topo_order

GBDT_DEEP = dict(max_leaf_nodes=31, min_samples_leaf=10)
GBDT_BIG = dict(max_leaf_nodes=63, min_samples_leaf=5, learning_rate=0.1, max_iter=300,
                l2_regularization=0.5)

# In priority order: the scheduler keeps a component only if all its folds and
# the refit still fit into the budget.
DEFAULT_COMPONENTS = [
    lambda: Marginal(),
    lambda: Backoff(depth=1),
    lambda: Backoff(depth=2),
    lambda: Backoff(depth=3),
    lambda: MultiTaskNet(hidden=0, l2=1e-2),
    lambda: MultiTaskNet(hidden=64, l2=3e-3, seed=0),
    lambda: LatentClass(K=8),
    lambda: FactorVAE(d=5),
    lambda: FactorVAE(d=3, seed=1),
    lambda: GBDT(),
    lambda: Direct(GBDT()),
    lambda: GBDT(params=GBDT_DEEP, name="gbdt_deep"),
    lambda: Direct(GBDT(params=GBDT_DEEP, name="gbdt_deep")),
    lambda: GBDT(params=GBDT_BIG, name="gbdt_big"),
    lambda: RandomForest(),
    lambda: TabPFN(max_items=1000, n_estimators=1, use_cap=False, name="tabpfn_fast"),
    lambda: ProfileMatch(),
    lambda: Direct(ProfileMatch()),
    lambda: MultiTaskNet(hidden=0, l2=1e-2, pairs=5),
    lambda: OrdinalLogit(),
    lambda: TabPFN(max_items=40),
    lambda: Direct(Backoff(depth=2)),
    lambda: Direct(MultiTaskNet(hidden=0, l2=1e-2)),
    lambda: Direct(LatentClass(K=8)),
    lambda: Direct(MultiTaskNet(hidden=64, l2=3e-3, seed=0)),
    lambda: MultiTaskNet(hidden=0, l2=1e-3),
    lambda: LatentClass(K=16, seed=1),
    lambda: MultiTaskNet(hidden=128, l2=3e-3, seed=1),
    lambda: MultiTaskNet(hidden=64, l2=3e-3, seed=2),
    lambda: MultiTaskNet(hidden=0, l2=3e-3, pairs=7),
]
SAFETY = 0.85            # share of the budget available for planned work
BLEND_RESERVE_S = 10.0


def _log(msg, verbose):
    if verbose:
        print(f"[simbench] {msg}", flush=True)


def _inner(model):
    return getattr(model, "inner", model)


def run(frame: pd.DataFrame, schema: dict, components=None, n_folds: int = 3,
        budget_s: float = 250.0, seed: int = 0, verbose: bool = False,
        return_info: bool = False):
    t0 = time.time()
    components = components or DEFAULT_COMPONENTS
    items, given, predict = parse_schema(schema)
    order = topo_order(items, predict)
    enc = Encoded(frame, items, given, predict)

    B = np.stack([enc.blank[j] for j in predict], axis=1) if predict \
        else np.zeros((enc.n, 0), bool)
    target_rows = np.nonzero(B.any(axis=1))[0]
    visible_rows = np.nonzero(~B.any(axis=1))[0]
    vis, tgt = enc.subset(visible_rows), enc.subset(target_rows)
    _log(f"{len(visible_rows)} visible, {len(target_rows)} to predict, "
         f"{len(predict)} items, {enc.X.shape[1]} features", verbose)

    if len(visible_rows) < 60 * n_folds:
        components = [lambda: Marginal()]
        n_folds = 2 if len(visible_rows) >= 4 else 1

    # ---- out-of-fold predictions -------------------------------------------
    fold = np.random.default_rng(seed).permutation(len(visible_rows)) % n_folds
    limit = budget_s * SAFETY
    HEAVY_ITEM_CAP[0] = 1000 if budget_s >= 1000 else 40
    set_deadline(t0 + limit)

    fold_data = []
    for f in range(n_folds):
        tr, va = np.nonzero(fold != f)[0], np.nonzero(fold == f)[0]
        if n_folds == 1:
            tr = va = np.arange(len(visible_rows))
        e_tr, e_va = vis.subset(tr), vis.subset(va)
        fold_data.append((va, e_tr, e_va, GateModel().fit(e_tr)))

    def compose(model, gates, cond, e):
        return cond if getattr(model, "direct", False) else gates.compose(cond, e, order)

    names, factories, oof, fold_avg, refit_est = [], [], {}, {}, {}
    refit_s = 0.0
    min_net_fold = None
    too_slow = set()
    for factory in components:
        proto = factory()
        name, cls = proto.name, type(_inner(proto)).__name__
        is_net = cls == "MultiTaskNet"
        if cls == "TabPFN" and cls in too_slow:
            _log(f"skip {name}: a cheaper TabPFN was already too slow", verbose)
            continue
        if is_net and min_net_fold is not None:
            lower = (time.time() - t0) + min_net_fold * (n_folds - 1) + refit_s \
                + min_net_fold * n_folds / max(1, n_folds - 1) + BLEND_RESERVE_S
            if lower > limit:
                _log(f"skip {name}: cannot fit even at {min_net_fold:.1f}s/fold", verbose)
                continue
        ts = time.time()
        full_oof = {j: np.zeros((vis.n, items[j].k_full)) for j in predict}
        full_tgt = {j: np.zeros((tgt.n, items[j].k_full)) for j in predict}
        kept = True
        for f, (va, e_tr, e_va, gates) in enumerate(fold_data):
            try:
                model = factory().fit(e_tr)
                full = compose(model, gates, model.predict(e_va), e_va)
                if tgt.n:
                    ft = compose(model, gates, model.predict(tgt), tgt)
                    for j in predict:
                        full_tgt[j] += ft[j] / n_folds
                del model
            except Exception as e:
                _log(f"drop {name}: {type(e).__name__}: {e}", verbose)
                kept = False
                break
            for j in predict:
                full_oof[j][va] = full[j]
            if f == 0:
                t_fold = time.time() - ts
                if is_net:
                    min_net_fold = t_fold if min_net_fold is None else min(min_net_fold, t_fold)
                t_refit = t_fold * n_folds / max(1, n_folds - 1)
                projected = (time.time() - t0) + t_fold * (n_folds - 1) + refit_s + t_refit \
                    + BLEND_RESERVE_S
                if projected > limit:
                    _log(f"drop {name}: fold took {t_fold:.1f}s, projected "
                         f"{projected:.0f}s > {limit:.0f}s", verbose)
                    too_slow.add(cls)
                    kept = False
                    break
        if not kept:
            continue
        refit_s += t_refit
        names.append(name)
        factories.append(factory)
        oof[name], fold_avg[name], refit_est[name] = full_oof, full_tgt, t_refit
        _log(f"OOF {name}: {time.time() - ts:.1f}s (elapsed {time.time() - t0:.0f}s)", verbose)

    # ---- blend ---------------------------------------------------------------
    set_deadline(t0 + budget_s)
    blender = Blender(names).fit(oof, vis.y_full)
    _log("OOF mean log score: " + ", ".join(f"{k}={v:.4f}" for k, v in blender.report.items()),
         verbose)
    _log("global weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, blender.global_w)),
         verbose)

    # ---- refit on all visible rows -------------------------------------------
    # A refit that would not finish in time falls back to the average of the
    # fold models, which is close in quality; a refit cut off by the deadline
    # would not be.
    t_refit0 = time.time()
    hard_end = t0 + budget_s - BLEND_RESERVE_S
    gates = GateModel().fit(vis)
    preds, n_fallback = {}, 0
    for i, (name, factory) in enumerate(zip(names, factories)):
        if blender.global_w[i] < 1e-4 and all(w[i] < 1e-3 for w in blender.w.values()):
            preds[name] = None
            continue
        if time.time() + refit_est[name] > hard_end:
            preds[name], n_fallback = fold_avg[name], n_fallback + 1
            continue
        try:
            model = factory().fit(vis)
            cond = model.predict(tgt)
            if time.time() > hard_end:
                preds[name], n_fallback = fold_avg[name], n_fallback + 1
                continue
            preds[name] = compose(model, gates, cond, tgt)
        except Exception as e:
            _log(f"refit {name} failed ({type(e).__name__}: {e})", verbose)
            preds[name], n_fallback = fold_avg[name], n_fallback + 1
    _log(f"refit done in {time.time() - t_refit0:.1f}s, {n_fallback} fold averages", verbose)
    live = [n for n in names if preds[n] is not None]
    if not live:
        raise RuntimeError("no component survived the refit")

    final = {}
    for j in predict:
        w = blender.w.get(j, blender.global_w)
        idx = [names.index(n) for n in live]
        ww = w[idx] / w[idx].sum()
        final[j] = normalize(blender.calibrate(sum(wi * preds[n][j] for wi, n in zip(ww, live)), j))

    # ---- post-processing, each step validated on the OOF blend ---------------
    oof_final = {j: normalize(blender.calibrate(blender.combine(
        {m: {j: oof[m][j]} for m in names}, j), j)) for j in predict}

    sub = SubtreeCalibrator().fit(items, predict, oof_final, vis.y_full)
    if sub.params:
        oof_final = sub.apply(oof_final)
        final = {j: normalize(p) for j, p in sub.apply(final).items()}

    families = find_families(schema, predict)
    if families:
        fam = FamilyAdjust().fit(families, oof_final, vis.y_full)
        final = {j: normalize(p) for j, p in fam.apply(final).items()}

    ordinal = find_ordinal(schema, predict)
    if ordinal:
        osm = OrdinalSmoother().fit(ordinal, oof_final, vis.y_full)
        final = {j: normalize(p) for j, p in osm.apply(final).items()}

    # ---- output --------------------------------------------------------------
    vectors = []
    for i, r in enumerate(target_rows):
        for j in predict:
            if enc.blank[j][r]:
                vectors.append(final[j][i].tolist())
    _log(f"done in {time.time() - t0:.1f}s, {len(vectors)} vectors", verbose)
    if return_info:
        return vectors, {"report": blender.report, "names": names, "T": blender.T,
                         "global_w": blender.global_w.tolist(), "seconds": time.time() - t0}
    return vectors

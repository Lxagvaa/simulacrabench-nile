"""SimulacraBench submission entry point: predict(frame, schema), called once
per instrument. All fitting happens inside predict() on the visible rows."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simbench.pipeline import run  # noqa: E402

# Total budget for all three instruments (limits: 900 s / 3600 s), with a margin.
BUDGET_BY_PHASE = {1: 780.0, 2: 3300.0}

# Relative compute need per instrument; the remaining time is split in
# proportion to the instruments still to come.
WEIGHTS = {"unicef": 1.0, "world_bank": 2.0, "unhcr": 3.2}
MARKERS = {"unhcr": "governorate_col", "world_bank": "user.age", "unicef": "Q01"}

_T_IMPORT = time.time()
_state = {"calls": 0, "total": None, "seen": set()}


def _instrument(schema):
    for name, marker in MARKERS.items():
        if marker in schema["items"]:
            return name
    return None


def _phase(frame, schema):
    """2 (final) when TRAIN and DEV are both visible, else 1."""
    pred = [n for n, r in schema["items"].items() if r["class"] == "PREDICT"]
    n_visible = int((~frame[pred].isna().any(axis=1)).sum()) if pred else 0
    split = schema.get("split", {})
    return 2 if n_visible >= split.get("n_train", 0) + split.get("n_dev", 0) > 0 else 1


def _budget(schema):
    remaining = _state["total"] - (time.time() - _T_IMPORT)
    inst = _instrument(schema)
    _state["seen"].add(inst)
    mean_w = sum(WEIGHTS.values()) / len(WEIGHTS)
    later = [w for k, w in WEIGHTS.items() if k not in _state["seen"]]
    later = later[: max(0, 3 - _state["calls"])]
    later += [mean_w] * max(0, 3 - _state["calls"] - len(later))
    w = WEIGHTS.get(inst, mean_w)
    return max(30.0, remaining * w / (w + sum(later)))


def predict(frame, schema):
    _state["calls"] += 1
    if _state["total"] is None:
        phase = _phase(frame, schema)
        _state["total"] = float(os.environ.get("SIMBENCH_BUDGET_S", BUDGET_BY_PHASE[phase]))
        _state["folds"] = 3 if phase == 1 else 5
    verbose = os.environ.get("SIMBENCH_VERBOSE") == "1"
    try:
        return run(frame, schema, budget_s=_budget(schema), n_folds=_state["folds"],
                   verbose=verbose)
    except Exception:
        if verbose:
            raise
        return _fallback(frame, schema)


def _fallback(frame, schema):
    """Crowd marginals, so that an unexpected error never fails the submission."""
    gv = schema.get("gated_value", "NA_GATED")
    items = [n for n, r in schema["items"].items() if r["class"] == "PREDICT"]
    marg = {}
    for n in items:
        opts = list(schema["items"][n]["values"]) + ([gv] if schema["items"][n].get("gate") else [])
        c = frame[n].astype(str).value_counts() if n in frame else {}
        w = np.array([c.get(str(o), 0) + 0.5 for o in opts], float)
        marg[n] = (w / w.sum()).tolist()
    blank = pd.DataFrame({n: frame[n].isna() if n in frame else True for n in items},
                         index=frame.index).to_numpy()
    return [marg[items[c]] for r in range(blank.shape[0]) for c in range(len(items))
            if blank[r, c]]

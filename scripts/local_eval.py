"""Fast in-process evaluation on sandbox data, mirroring score.py.

    python scripts/local_eval.py --kit ../sb --instruments unicef world_bank unhcr --phase 1

Reports per-instrument skill for our submission and for the marginal baseline,
OOF diagnostics, and wall-clock time. Use the official score.py for the final
check (fresh venv, network cut): see scripts/official_check.sh.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submission"))
from simbench.pipeline import run  # noqa: E402
from simbench.schema_utils import parse_schema  # noqa: E402

PHASES = {1: (("TRAIN",), "DEV"), 2: (("TRAIN", "DEV"), "TEST")}
FLOOR = 1e-3


def options_for(schema, name):
    rec = schema["items"][name]
    return list(rec["values"]) + ([schema["gated_value"]] if rec.get("gate") else [])


def build(kit, inst, phase):
    data = os.path.join(kit, "_sandbox", inst)
    schema = json.load(open(os.path.join(data, "schema.json")))
    schema.setdefault("gated_value", "NA_GATED")
    df = pd.read_parquet(os.path.join(data, "respondents.parquet")).astype(object)
    items = [n for n, r in schema["items"].items() if r["class"] in ("GIVEN", "PREDICT")]
    scored = [n for n, r in schema["items"].items() if r["class"] == "PREDICT"]
    vis_roles, hid_role = PHASES[phase]
    vis = df[df["role"].isin(vis_roles)][["respondent_id"] + items].reset_index(drop=True)
    hid = df[df["role"] == hid_role][["respondent_id"] + items].reset_index(drop=True)
    blank = hid.copy()
    blank[scored] = np.nan
    frame = pd.concat([vis, blank], ignore_index=True)
    truth = [(n, hid[n].iloc[r]) for r in range(len(hid)) for n in scored]
    return frame, schema, truth, scored


def skill(vectors, truth, schema, scored):
    U = np.mean([np.log(len(options_for(schema, n))) for n in scored])
    ls = []
    for v, (n, y) in zip(vectors, truth):
        p = np.asarray(v, float)
        p = p / p.sum()
        p = FLOOR + (1 - p.size * FLOOR) * p
        ls.append(np.log(p[options_for(schema, n).index(y)]))
    return 1 + np.mean(ls) / U


def marginal_vectors(frame, schema):
    sys.path.insert(0, os.path.join(ARGS.kit, "baseline", "marginal_counts"))
    import importlib
    m = importlib.import_module("main")
    return m.predict(frame, schema)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=os.path.join(ROOT, "starter_kit"))
    ap.add_argument("--instruments", nargs="+", default=["unicef", "world_bank", "unhcr"])
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--budget", type=float, default=250.0, help="per-instrument seconds")
    ARGS = ap.parse_args()
    rows, total = [], 0.0
    for inst in ARGS.instruments:
        frame, schema, truth, scored = build(ARGS.kit, inst, ARGS.phase)
        t = time.time()
        vecs, info = run(frame, schema, budget_s=ARGS.budget, verbose=not ARGS.quiet, return_info=True)
        dt = time.time() - t
        total += dt
        assert len(vecs) == len(truth), (len(vecs), len(truth))
        s_ours = skill(vecs, truth, schema, scored)
        s_base = skill(marginal_vectors(frame, schema), truth, schema, scored)
        rows.append((inst, s_base, s_ours, dt))
    print("\ninstrument      marginal   ours    delta   seconds")
    for inst, b, o, dt in rows:
        print(f"{inst:14s} {b:8.4f} {o:8.4f} {o - b:+8.4f} {dt:8.1f}")
    print(f"{'MEAN':14s} {np.mean([r[1] for r in rows]):8.4f} {np.mean([r[2] for r in rows]):8.4f}"
          f"          total {total:.1f}s")

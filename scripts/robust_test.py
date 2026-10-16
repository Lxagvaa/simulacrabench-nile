"""Adversarial frames the real data could contain: NaN in GIVEN cells,
off-schema values in visible cells, numeric option values. The output must
always have exactly one vector per NaN PREDICT cell, of the right width.

    uv run python scripts/robust_test.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submission"))
import main  # noqa: E402
from simbench.pipeline import run  # noqa: E402


def build(inst):
    d = os.path.join(ROOT, "starter_kit", "_sandbox", inst)
    schema = json.load(open(f"{d}/schema.json"))
    schema.setdefault("gated_value", "NA_GATED")
    df = pd.read_parquet(f"{d}/respondents.parquet").astype(object)
    items = [n for n, r in schema["items"].items() if r["class"] in ("GIVEN", "PREDICT")]
    pred = [n for n, r in schema["items"].items() if r["class"] == "PREDICT"]
    vis = df[df.role == "TRAIN"][["respondent_id"] + items].reset_index(drop=True)
    hid = df[df.role == "DEV"][["respondent_id"] + items].reset_index(drop=True)
    hid[pred] = np.nan
    return pd.concat([vis, hid], ignore_index=True), schema, pred


def check(label, got, want):
    print(f"{label:16s} {got} expected {want}  {'OK' if got == want else 'FAIL'}")
    return got == want


frame, schema, pred = build("unicef")
n_expected = int(frame[pred].isna().sum().sum())
given = [n for n, r in schema["items"].items() if r["class"] == "GIVEN"]
ok = True

f2 = frame.copy()
f2.loc[f2.index[-50:], given[0]] = np.nan                 # NaN in held-out GIVEN cells
f2.loc[0, pred[0]] = "SOMETHING_NOT_IN_SCHEMA"            # off-schema visible value
ok &= check("adversarial run", len(run(f2, schema, budget_s=40)), n_expected)
ok &= check("fallback", len(main._fallback(f2, schema)), n_expected)

s3, f3 = json.loads(json.dumps(schema)), frame.copy()     # numeric options, float cells
for c in given + pred:
    vals = schema["items"][c]["values"]
    m = {o: i + 1 for i, o in enumerate(vals)}
    s3["items"][c]["values"] = [m[o] for o in vals]
    f3[c] = f3[c].map(lambda x, m=m: float(m[x]) if x in m else x)
v3 = run(f3, s3, budget_s=40)
ok &= check("numeric run", len(v3), n_expected)
cells = [j for r in range(len(f3)) for j in pred if pd.isna(f3.loc[r, j])]
widths = all(len(x) == len(s3["items"][j]["values"]) + bool(s3["items"][j].get("gate"))
             for x, j in zip(v3, cells))
print(f"{'widths':16s} {'OK' if widths else 'FAIL'}")
sys.exit(0 if ok and widths else 1)

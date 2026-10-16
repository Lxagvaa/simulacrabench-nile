"""Run instruments in ONE process through main.predict, like the hosted run:
shared time budget, size-weighted split, main's fallback in place.

    uv run python scripts/eval_main.py submission unhcr,world_bank,unicef [phase]

The first argument is a submission directory (e.g. an unzipped dist/*.zip);
instrument names are sandbox directories under starter_kit/_sandbox/.
"""
import importlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sub = os.path.abspath(sys.argv[1])
if not os.path.isfile(os.path.join(sub, "main.py")):
    sys.exit(f"no main.py in {sub}")
order = sys.argv[2].split(",")
PHASE = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # 2 = final: TRAIN+DEV visible, TEST scored
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import local_eval as le  # noqa: E402

le.ARGS = type("A", (), {"kit": os.path.join(ROOT, "starter_kit")})()
data = {i: le.build(le.ARGS.kit, i, PHASE) for i in order}
sys.path.insert(0, sub)
for m in [k for k in sys.modules if k == "main" or k.startswith("simbench")]:
    del sys.modules[m]
t0 = time.time()
main = importlib.import_module("main")
assert os.path.dirname(os.path.abspath(main.__file__)) == sub, main.__file__
for i in order:
    frame, schema, truth, scored = data[i]
    t = time.time()
    v = main.predict(frame, schema)
    assert len(v) == len(truth), (i, len(v), len(truth))
    print(f"{i:13s} skill {le.skill(v, truth, schema, scored):.4f}  {time.time() - t:6.1f}s",
          flush=True)
print(f"total {time.time() - t0:.0f}s")

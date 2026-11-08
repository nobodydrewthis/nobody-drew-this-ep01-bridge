"""Rebuild generation 0 of the episode run exactly (bridge40, seed 1: the same 160 random_genome calls on the same
generator as evolve()) and export all 160 founders for the '160 random bridges' shot -> motion/public/ndt/ep01/gen0.json.
Check: every founder's mass must equal its lineage fitness (the run's own record), so the picture is the real generation 0.
Usage: py -I export_gen0.py [run_dir] [out.json]"""
import json
import os
import sys

for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ndt_engine as E
from briefs import ALL
from run_candidates import POP, calibrate_joint

HERE = os.path.dirname(os.path.abspath(__file__))
# defaults: the studio layout; in the release: python export_gen0.py ../run_bridge40_seed1 generation0_rebuilt.json
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "run_bridge40_s1")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "data", "gen0.json")
brief = ALL["bridge40"]()
calibrate_joint(brief)
lin = np.load(os.path.join(DATA, "lineage.npz"))
fit = dict(zip(lin["id"].tolist(), lin["fit"].tolist()))
anc = json.load(open(os.path.join(DATA, "ancestry.json")))
founder = anc[0]["id"]
rng = np.random.default_rng(1)
out, bad = [], 0
for i in range(POP):
    g = E.random_genome(brief, rng)
    f, A, S = E.evaluate(brief, g)
    if abs(f - fit[i]) > 1e-9:
        bad += 1
    d = {"id": i, "n": np.asarray(S.nodes).round(3).tolist(), "b": np.asarray(S.bars).tolist(), "ok": bool(f < E.INFEAS)}
    if A is not None:
        d["m"] = round(float(f), 4)
        d["D"] = [round(float(x), 4) for x in brief.steel.diameter(A)]
    out.append(d)
json.dump({"gen": 0, "founder": founder, "designs": out}, open(OUT, "w"), separators=(",", ":"))
ok = sum(d["ok"] for d in out)
print(f"generation 0 rebuilt: {len(out)} designs, {ok} pass every check, {len(out) - ok} fail; "
      f"fitness mismatches vs the lineage: {bad}; winner's founder id {founder} "
      f"({out[founder].get('m')} t vs ancestry {anc[0]['fit']:.4f})")

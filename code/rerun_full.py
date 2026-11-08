"""Re-run the chosen brief and seed with full logging for the visuals (population samples every generation and
population membership), and check it reproduces the candidate run exactly.
Usage: py -I rerun_full.py <brief> <seed> <candidate_json> <outdir>"""
import os
import sys

for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import ndt_engine as E
from briefs import ALL
from run_candidates import below_normal, calibrate_joint, POP, ELITE, TOUR

below_normal()
name, seed, cand, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(outdir, exist_ok=True)
ref = json.load(open(cand))
brief = ALL[name]()
calibrate_joint(brief)
gens = len(ref["hist"])
r = E.evolve(brief, seed=seed, pop=POP, gens=gens, elite=ELITE, tour=TOUR, sample_every=1, sample_n=48,
             log=lambda m: print(m, flush=True))
same = abs(r["fitness"] - ref["analysis"]["mass_t"]) < 1e-9 or abs(r["fitness"] - ref["raw"]["mass_t"]) < 1e-9
print("reproduced:", same, r["fitness"], ref["raw"]["mass_t"], "winner id", r["winner_id"])
lid, lpar, lgen, lfit = r["lineage"]
np.savez_compressed(os.path.join(outdir, "lineage.npz"), id=lid, parent=lpar, gen=lgen, fit=lfit,
                    popids=np.array(r["popids"], np.int32))


def pheno(g):
    S = E.decode(brief, g)
    if not S.ok:
        return None
    A, ok = S.size()
    d = {"n": S.nodes.round(3).tolist(), "b": S.bars.tolist()}
    if ok:
        d["d"] = [round(float(x), 4) for x in brief.steel.diameter(A)]
        d["m"] = round(S.mass(A), 4)
    return d


samples = []
for s in r["samples"]:
    row = {"gen": s["gen"], "designs": []}
    for i in s["ids"]:
        g = r["genomes"].get(i)
        if g is not None:
            p = pheno(g)
            if p:
                p["id"] = int(i)
                p["fit"] = float(lfit[i])
                row["designs"].append(p)
    samples.append(row)
json.dump(samples, open(os.path.join(outdir, "samples.json"), "w"), separators=(",", ":"))
chain = E.ancestry(lpar, r["winner_id"])
anc = []
for i in chain:
    g = r["genomes"].get(i)
    if g is None:
        continue
    p = pheno(g)
    if p:
        p["id"] = int(i)
        p["gen"] = int(lgen[i])
        p["fit"] = float(lfit[i])
        anc.append(p)
json.dump(anc, open(os.path.join(outdir, "ancestry.json"), "w"), separators=(",", ":"))
json.dump({"brief": name, "seed": seed, "winner_id": r["winner_id"], "fitness": r["fitness"], "reproduced": same,
           "generations": gens, "evals": int(len(lid)), "seconds": r["seconds"], "hist": r["hist"]},
          open(os.path.join(outdir, "run.json"), "w"), indent=1)
print("samples", len(samples), "ancestry", len(anc))

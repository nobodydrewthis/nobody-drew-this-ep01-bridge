"""Run every candidate brief with two seeds in parallel (below-normal priority), then analyse winners against
the textbook designs. Usage: py -I run_candidates.py <outdir> [workers] [brief,brief,...]"""
import os
import sys

for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import time
import multiprocessing as mp
import numpy as np

POP, GENS, ELITE, TOUR, TLIM = int(os.environ.get("NDT_POP", 160)), int(os.environ.get("NDT_GENS", 400)), 4, 3, 1500


def below_normal():
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def calibrate_joint(brief):
    """Disclosed rule: every joint weighs 12% of the textbook design's member steel divided by its joints."""
    import ndt_engine as E
    for _ in range(3):
        S = E.textbook_structure(brief)
        A, ok = S.size()
        brief.m_joint = 0.12 * S.member_mass(A) / S.n
    return brief.m_joint


def analyse(brief, S, A, T, At):
    """Compare a design S (areas A) with the textbook T (areas At)."""
    import ndt_engine as E
    out = {}
    m_e, m_t = S.mass(A), T.mass(At)
    out["mass_t"] = m_e
    out["textbook_mass_t"] = m_t
    out["mass_ratio"] = m_e / m_t
    # equal mass: scale the member areas of S so that its total mass equals the textbook's
    s = (m_t - S.n * brief.m_joint) / S.member_mass(A)
    Aeq = A * s
    out["equal_mass_scale"] = s
    rows = []
    for c in brief.cases:
        le, lt = S.first_fail(Aeq, c), T.first_fail(At, c)
        rows.append({"case": c["name"], "lam_evolved": le, "lam_textbook": lt})
    out["equal_mass_cases"] = rows
    out["beats_all_cases"] = all(r["lam_evolved"] > r["lam_textbook"] for r in rows)
    out["worst_case_ratio"] = min(r["lam_evolved"] / r["lam_textbook"] for r in rows)
    un = []
    for c in brief.unseen:
        ev_e, pk_e = S.ramp(Aeq, c, max_events=80)
        ev_t, pk_t = T.ramp(At, c, max_events=80)
        un.append({"case": c["name"], "first_evolved": ev_e[0]["lam"], "collapse_evolved": pk_e,
                   "first_textbook": ev_t[0]["lam"], "collapse_textbook": pk_t})
    out["unseen"] = un
    # look: share of evolved bars that coincide with a textbook bar (both ends within 0.03 * size)
    tol = 0.03 * brief.size
    TP = np.stack([T.nodes[T.bars[:, 0]], T.nodes[T.bars[:, 1]]], 1)
    shared = 0
    for a, b in S.bars:
        p, q = S.nodes[a], S.nodes[b]
        d1 = np.hypot(*(TP[:, 0] - p).T) + np.hypot(*(TP[:, 1] - q).T)
        d2 = np.hypot(*(TP[:, 0] - q).T) + np.hypot(*(TP[:, 1] - p).T)
        if (np.minimum(d1, d2) < 2 * tol).any():
            shared += 1
    out["shared_bars"] = shared / max(S.m, 1)
    out["bars"], out["joints"] = int(S.m), int(S.n)
    out["violations"] = E.violations(brief, S)
    return out


def run_one(job):
    below_normal()
    import ndt_engine as E
    from briefs import ALL
    name, seed, outdir = job
    brief = ALL[name]()
    mj = calibrate_joint(brief)
    logf = open(os.path.join(outdir, f"{name}_s{seed}.log"), "w", encoding="utf-8")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()
    log(f"{name} seed {seed}: joint mass {mj * 1000:.3f} kg")
    r = E.evolve(brief, seed=seed, pop=POP, gens=GENS, elite=ELITE, tour=TOUR, log=log, time_limit=TLIM)
    f, A, S = E.evaluate(brief, r["winner"])
    raw = {"mass_t": f, "bars": int(S.m), "joints": int(S.n)}
    C, CA, removed = E.cleanup(brief, S, A)
    T = E.textbook_structure(brief)
    At, ok = T.size()
    res = {"brief": name, "title": brief.title, "seed": seed, "seconds": r["seconds"], "evals": len(r["lineage"][0]),
           "joint_mass_t": mj, "raw": raw, "cleanup_removed": removed,
           "analysis": analyse(brief, C, CA, T, At), "textbook_name": brief.textbook["name"],
           "hist": r["hist"], "winner": C.to_json(CA), "textbook": T.to_json(At),
           "steel": brief.steel.to_json()}
    lid, lpar, lgen, lfit = r["lineage"]
    np.savez_compressed(os.path.join(outdir, f"{name}_s{seed}_lineage.npz"), id=lid, parent=lpar, gen=lgen, fit=lfit)
    chain = E.ancestry(lpar, r["winner_id"])
    anc = []
    for i in chain:
        g = r["genomes"].get(i)
        if g is None:
            continue
        S_ = E.decode(brief, g)
        A_, ok_ = S_.size() if S_.ok else (None, False)
        anc.append({"id": int(i), "gen": int(lgen[i]), "fit": float(lfit[i]), "nodes": S_.nodes.round(4).tolist(),
                    "bars": S_.bars.tolist(), "A": None if A_ is None else [float(x) for x in A_]})
    res["ancestry_len"] = len(anc)
    json.dump(E.jsonable(res), open(os.path.join(outdir, f"{name}_s{seed}.json"), "w"), indent=1)
    json.dump(anc, open(os.path.join(outdir, f"{name}_s{seed}_ancestry.json"), "w"))
    a = res["analysis"]
    log(f"DONE {name} s{seed}: {a['mass_t']:.3f} t vs textbook {a['textbook_mass_t']:.3f} t "
        f"({100 * (a['mass_ratio'] - 1):+.1f}%), beats all cases {a['beats_all_cases']}, "
        f"shared bars {100 * a['shared_bars']:.0f}%, {r['seconds']:.0f}s")
    logf.close()
    return name, seed, a["mass_ratio"], a["beats_all_cases"], a["shared_bars"], r["seconds"]


if __name__ == "__main__":
    outdir = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    from briefs import ALL
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ALL)
    os.makedirs(outdir, exist_ok=True)
    below_normal()
    jobs = [(n, s, outdir) for s in (1, 2) for n in names]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            n, s, mr, beats, sh, sec = res
            print(f"{time.time() - t0:6.0f}s  {n:13s} s{s}: mass {100 * (mr - 1):+6.1f}%  beats all {beats}  "
                  f"shared {100 * sh:3.0f}%  ({sec:.0f}s)", flush=True)
    print("all done", time.time() - t0)

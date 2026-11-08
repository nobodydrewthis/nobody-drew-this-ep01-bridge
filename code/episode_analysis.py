"""Episode 1 numbers: the evolved bridge (bridge40, seed 1) against the Pratt truss at its own best depth.
Writes ../data/episode.json with everything the video shows (masses, equal-steel ramps, failure order, unseen load),
and the geometry Blender rebuilds. Also runs an independent rigid-jointed frame solve (3 DOF per joint, bending
included) as an in-house second solver.
Usage: py -I episode_analysis.py"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ndt_engine as E
import briefs as BR
from run_candidates import calibrate_joint
from fair_textbook import pratt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
G = 9.81

b = BR.bridge40()
mj = calibrate_joint(b)          # joint weight fixed from the standard 5 m Pratt (as in the candidate run)
res = json.load(open(os.path.join(DATA, "candidates", "bridge40_s1.json")))
W = res["winner"]

# evolved structure (rebuild in brief node space)
nodes = np.array(W["nodes"])
fixedmask = np.array(W["fixed"])
orig = np.array(W["orig"])
full = np.vstack([b.fixed, nodes[~fixedmask]])
remap = np.empty(len(nodes), int)
remap[fixedmask] = orig[fixedmask]
remap[~fixedmask] = b.nf + np.arange((~fixedmask).sum())
S = E.Structure(b, full, remap[np.array(W["bars"])])
A, ok = S.size()
assert ok
print("violations evolved:", E.violations(b, S))

# Pratt at every depth -> best
best = None
for h in np.arange(3.0, 8.01, 0.25):
    nd, br = pratt(b, h)
    T = E.Structure(b, np.array(nd), np.array(br))
    At, okt = T.size()
    if okt and (best is None or T.mass(At) < best[2]):
        best = (h, T, T.mass(At), At)
h_best, T, m_t, At = best
nd5, br5 = pratt(b, 5.0)
T5 = E.Structure(b, np.array(nd5), np.array(br5))
A5, _ = T5.size()
m_e = S.mass(A)
print(f"evolved {m_e:.3f} t | Pratt best depth {h_best:.2f} m {m_t:.3f} t | Pratt 5 m {T5.mass(A5):.3f} t")

# equal steel: scale evolved member areas to the best Pratt's total mass
s = (m_t - S.n * b.m_joint) / S.member_mass(A)
Aeq = A * s
print(f"equal steel: evolved areas x{s:.3f} -> {S.mass(Aeq):.3f} t")

live_t_per_lam = 2 * 40e3 * 40 / G / 1000     # live load on both trusses (40 kN/m per truss over 40 m), tonnes per lambda


def ramp_report(Sx, Ax, case, label):
    ev, peak = Sx.ramp(Ax, case, max_events=120)
    out = []
    for e in ev:
        if e["bar"] < 0:
            out.append({"event": "mechanism", "lam": e["lam"]})
            continue
        a, bb = Sx.bars[e["bar"]]
        p, q = Sx.nodes[a], Sx.nodes[bb]
        out.append({"bar": int(e["bar"]), "lam": e["lam"], "mode": e["mode"], "N_kN": e["N"] / 1e3,
                    "from": p.round(3).tolist(), "to": q.round(3).tolist(), "cascade": e.get("cascade", False)})
    return {"case": case["name"], "design": label, "first_lam": ev[0]["lam"], "collapse_lam": peak, "events": out}


rows = []
for c in b.cases:
    re_ = ramp_report(S, Aeq, c, "evolved")
    rt_ = ramp_report(T, At, c, "pratt")
    rows.append({"case": c["name"], "evolved": re_, "pratt": rt_,
                 "ratio_first": re_["first_lam"] / rt_["first_lam"]})
    print(f"{c['name']:20s} first failure: evolved x{re_['first_lam']:.3f} ({re_['events'][0].get('mode')}) "
          f"vs Pratt x{rt_['first_lam']:.3f} ({rt_['events'][0].get('mode')}) -> x{re_['first_lam'] / rt_['first_lam']:.2f}")
un = []
for c in b.unseen:
    re_ = ramp_report(S, Aeq, c, "evolved")
    rt_ = ramp_report(T, At, c, "pratt")
    un.append({"case": c["name"], "evolved": re_, "pratt": rt_})
    print(f"UNSEEN {c['name']}: evolved first x{re_['first_lam']:.3f} collapse x{re_['collapse_lam']:.3f} | "
          f"Pratt first x{rt_['first_lam']:.3f} collapse x{rt_['collapse_lam']:.3f}")


# --------------------------------------------------------------------------- independent second solver: rigid-jointed frame
def frame_forces(Sx, Ax, F2):
    """2D frame (axial + bending, rigid joints), circular hollow sections. F2: (ndof_truss,) nodal forces.
    Returns axial forces (tension +) and the max bending stress per member."""
    st = b.steel
    n, m = Sx.n, Sx.m
    nd = 3 * n
    K = np.zeros((nd, nd))
    I = st.kI * Ax ** 2
    D = st.diameter(Ax)
    for k in range(m):
        i, j = Sx.bars[k]
        L, c, s_ = Sx.L[k], Sx.c[k], Sx.s[k]
        EA, EI = st.E * Ax[k], st.E * I[k]
        kl = np.array([[EA / L, 0, 0, -EA / L, 0, 0],
                       [0, 12 * EI / L ** 3, 6 * EI / L ** 2, 0, -12 * EI / L ** 3, 6 * EI / L ** 2],
                       [0, 6 * EI / L ** 2, 4 * EI / L, 0, -6 * EI / L ** 2, 2 * EI / L],
                       [-EA / L, 0, 0, EA / L, 0, 0],
                       [0, -12 * EI / L ** 3, -6 * EI / L ** 2, 0, 12 * EI / L ** 3, -6 * EI / L ** 2],
                       [0, 6 * EI / L ** 2, 2 * EI / L, 0, -6 * EI / L ** 2, 4 * EI / L]])
        Tm = np.zeros((6, 6))
        R = np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]])
        Tm[:3, :3] = R
        Tm[3:, 3:] = R
        kg = Tm.T @ kl @ Tm
        idx = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(idx, idx)] += kg
    F = np.zeros(nd)
    F[0::3] = F2[0::2]
    F[1::3] = F2[1::2]
    fixed = [3 * (d // 2) + (d % 2) for d in Sx.fixed_dofs]
    free = np.setdiff1d(np.arange(nd), fixed)
    u = np.zeros(nd)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    Nax = np.zeros(m)
    sb = np.zeros(m)
    for k in range(m):
        i, j = Sx.bars[k]
        L, c, s_ = Sx.L[k], Sx.c[k], Sx.s[k]
        R = np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]])
        ui = R @ u[3 * i:3 * i + 3]
        uj = R @ u[3 * j:3 * j + 3]
        Nax[k] = st.E * Ax[k] / L * (uj[0] - ui[0])
        EI = st.E * I[k]
        # end moments of a beam element from local displacements
        v1, t1, v2, t2 = ui[1], ui[2], uj[1], uj[2]
        M1 = EI / L ** 2 * (6 * v1 + 4 * L * t1 - 6 * v2 + 2 * L * t2)
        M2 = EI / L ** 2 * (6 * v1 + 2 * L * t1 - 6 * v2 + 4 * L * t2)
        sb[k] = max(abs(M1), abs(M2)) / (I[k] / (D[k] / 2))
    return Nax, sb


second = []
for label, Sx, Ax in (("evolved", S, Aeq), ("pratt", T, At)):
    for c in b.cases:
        sw = Sx.selfweight(Ax)
        Fd, Fl = Sx.case_loads(c)
        F2 = b.gam_d * Fd + b.gam_l * Fl + b.gam_s * sw
        Ntr, _ = Sx.forces(Ax, F2)
        Nfr, sb = frame_forces(Sx, Ax, F2)
        big = np.abs(Ntr) > 0.05 * np.abs(Ntr).max()
        dev = np.max(np.abs(Nfr[big] - Ntr[big]) / np.abs(Ntr[big]))
        # first-failure load factor with rigid joints: axial capacity check unchanged, bending stress added
        Nt, Nc = b.steel.caps(Ax, Sx.L)
        util_tr = np.where(Ntr >= 0, Ntr / Nt, -Ntr / Nc)
        util_fr = np.where(Nfr >= 0, Nfr / Nt, -Nfr / Nc) + sb / b.steel.fy
        second.append({"design": label, "case": c["name"], "max_axial_diff": float(dev),
                       "max_util_pinned": float(util_tr.max()), "max_util_rigid_with_bending": float(util_fr.max()),
                       "max_bending_stress_MPa": float(sb.max() / 1e6)})
        print(f"2nd solver {label:8s} {c['name']:20s} axial diff {100 * dev:5.2f}%  util pinned {util_tr.max():.3f} "
              f"rigid+bending {util_fr.max():.3f}")


def geo(Sx, Ax):
    return {"nodes": Sx.nodes.round(4).tolist(), "bars": Sx.bars.tolist(), "D_m": [float(x) for x in b.steel.diameter(Ax)],
            "A_m2": [float(x) for x in Ax], "t_m": [float(x) / b.steel.Dt for x in b.steel.diameter(Ax)],
            "supports": [[float(x) for x in b.fixed[i]] for i in b.supports]}


out = {"brief": "bridge40", "seed": 1, "joint_mass_t": mj, "pratt_best_depth_m": float(h_best),
       "mass_evolved_t": m_e, "mass_pratt_best_t": m_t, "mass_pratt_5m_t": T5.mass(A5),
       "saving_vs_best_pratt": 1 - m_e / m_t, "equal_steel_scale": float(s), "live_t_per_lambda": live_t_per_lam,
       "cases": rows, "unseen": un, "second_solver": second,
       "geometry": {"evolved_design": geo(S, A), "evolved_equal_steel": geo(S, Aeq), "pratt": geo(T, At)},
       "counts": {"evolved_members": int(S.m), "evolved_joints": int(S.n), "pratt_members": int(T.m), "pratt_joints": int(T.n)}}
json.dump(E.jsonable(out), open(os.path.join(DATA, "episode.json"), "w"), indent=1)
print("wrote episode.json")

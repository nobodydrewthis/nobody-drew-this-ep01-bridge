"""Progressive ramp to collapse, a step past the protocol's first-failure ramp (which is unchanged):
- a member that buckles is not deleted: it keeps its tension capacity (it can still pull) but carries no compression;
- a member that yields in tension is deleted (no strain hardening credited);
- live load ramps (dead load and self-weight held at 1.0) until the structure becomes a mechanism.
Used for what happens after the first failure in the break test. Writes ../data/progressive.json.
Usage: py -I ramp_progressive.py"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ndt_engine as E
import briefs as BR
from run_candidates import calibrate_joint

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
b = BR.bridge40()
calibrate_joint(b)
ep = json.load(open(os.path.join(DATA, "episode.json")))
st = b.steel


def build(geo):
    nodes = np.array(geo["nodes"])
    bars = np.array(geo["bars"])
    A = np.array(geo["A_m2"])
    return nodes, bars, A


def solve_set(nodes, bars, A, active, F, fixed):
    n = len(nodes)
    nd = 2 * n
    K = np.zeros((nd, nd))
    d = nodes[bars[:, 1]] - nodes[bars[:, 0]]
    L = np.linalg.norm(d, axis=1)
    c, s = d[:, 0] / L, d[:, 1] / L
    for k in np.flatnonzero(active):
        i, j = bars[k]
        B = np.array([-c[k], -s[k], c[k], s[k]])
        idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(idx, idx)] += st.E * A[k] / L[k] * np.outer(B, B)
    free = np.setdiff1d(np.arange(nd), fixed)
    Kf = K[np.ix_(free, free)]
    try:
        Lc = np.linalg.cholesky(Kf)
        if (np.diag(Lc) ** 2 / np.maximum(np.diag(Kf), 1e-30)).min() < 1e-9:
            return None
    except np.linalg.LinAlgError:
        return None
    u = np.zeros((nd,) + F.shape[1:])
    u[free] = np.linalg.solve(Kf, F[free])
    du = np.stack([u[2 * bars[:, 1]] - u[2 * bars[:, 0]], u[2 * bars[:, 1] + 1] - u[2 * bars[:, 0] + 1]], 1)
    ext = du[:, 0] * c[:, None] + du[:, 1] * s[:, None] if F.ndim > 1 else du[:, 0] * c + du[:, 1] * s
    N = (st.E * A / L)[:, None] * ext if F.ndim > 1 else st.E * A / L * ext
    return N, ext


def ramp(geo, case, label):
    nodes, bars, A = build(geo)
    m = len(bars)
    n = len(nodes)
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    Nt, Nc = st.caps(A, L)
    fixed = []
    for sx, sy, fx, fy in [(0.0, 0.0, True, True), (40.0, 0.0, False, True)]:
        i = int(np.argmin(np.hypot(nodes[:, 0] - sx, nodes[:, 1] - sy)))
        if fx:
            fixed.append(2 * i)
        if fy:
            fixed.append(2 * i + 1)
    # loads: dead (deck + self-weight + joints) and live (left-half traffic), per truss, characteristic
    Fd = np.zeros(2 * n)
    Fl = np.zeros(2 * n)
    for i, (x, y) in enumerate(nodes):
        if abs(y) < 1e-6:
            trib = 2.0 if x in (0.0, 40.0) else 4.0
            Fd[2 * i + 1] -= 30e3 * trib
            if x < 20 - 1e-6:
                Fl[2 * i + 1] -= 40e3 * trib
            elif abs(x - 20) < 1e-6:
                Fl[2 * i + 1] -= 40e3 * trib * 0.5
    w = st.rho * 9.81 * A * L
    for k, (i, j) in enumerate(bars):
        Fd[2 * i + 1] -= w[k] / 2
        Fd[2 * j + 1] -= w[k] / 2
    Fd[1::2] -= ep["joint_mass_t"] * 1000 * 9.81
    alive = np.ones(m, bool)
    buckled = np.zeros(m, bool)
    lam = 0.0
    events = []
    for _ in range(200):
        # active set: buckled members only while in tension
        active = alive.copy()
        for _it in range(30):
            res = solve_set(nodes, bars, A, active, np.stack([Fd + lam * Fl, Fl], 1), fixed)
            if res is None:
                break
            N2, _ = res
            Nnow = N2[:, 0]
            change = False
            for k in np.flatnonzero(alive & buckled):
                if active[k] and Nnow[k] < -1e-3:
                    active[k] = False
                    change = True
            if not change:
                break
        if res is None:
            events.append({"event": "mechanism", "lam": lam})
            break
        N2, ext = res
        # N(lam') = Nnow + (lam' - lam) * Nl  for the current active set
        Nnow, Nl = N2[:, 0], N2[:, 1]
        best, kind, kbest = np.inf, None, -1
        for k in np.flatnonzero(active):
            cands = []
            if Nl[k] > 1e-9:
                cands.append(((Nt[k] - Nnow[k]) / Nl[k], "yield"))
            if Nl[k] < -1e-9 and not buckled[k]:
                cands.append(((Nnow[k] + Nc[k]) / (-Nl[k]), "buckle"))
            if Nl[k] < -1e-9 and buckled[k]:
                cands.append(((Nnow[k]) / (-Nl[k]), "slack"))
            for dl, kd in cands:
                if dl >= -1e-9 and dl < best:
                    best, kind, kbest = max(dl, 0.0), kd, k
        if not np.isfinite(best):
            events.append({"event": "no further failure", "lam": lam})
            break
        lam += best
        a_, b_ = bars[kbest]
        if kind == "buckle":
            buckled[kbest] = True
            events.append({"event": "buckle", "bar": int(kbest), "lam": lam, "from": nodes[a_].tolist(), "to": nodes[b_].tolist()})
        elif kind == "yield":
            alive[kbest] = False
            events.append({"event": "yield", "bar": int(kbest), "lam": lam, "from": nodes[a_].tolist(), "to": nodes[b_].tolist()})
        else:
            pass   # a buckled member going slack: handled by the active set next round
        if lam > 20:
            break
    return events


out = {}
case = "traffic left half"
for key, label in (("pratt", "pratt"), ("evolved_equal_steel", "evolved")):
    ev = ramp(ep["geometry"][key], case, label)
    out[label] = ev
    live = ep["live_t_per_lambda"] / 2
    print(label, [(e["event"], e.get("bar"), round(e["lam"], 3), round(e["lam"] * live, 1)) for e in ev[:8]])
json.dump(out, open(os.path.join(DATA, "progressive.json"), "w"), indent=1)

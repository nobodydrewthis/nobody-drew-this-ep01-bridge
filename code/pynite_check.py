"""Gate 1 cross-check with an external solver (PyNite 3.2, MIT). Same geometry, sections and nodal loads as our engine.
1. Pinned model (member ends released, like ours): member forces must match.
2. Rigid model (welded joints, bending included): does the evolved design still beat the Pratt at equal steel?
PyNite reports tension as negative; ours is tension positive.
Run with: projects/channel4/.venv/Scripts/python.exe -I pynite_check.py"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from Pynite import FEModel3D
import ndt_engine as E
import briefs as BR
from run_candidates import calibrate_joint
from fair_textbook import pratt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
b = BR.bridge40()
calibrate_joint(b)
ep = json.load(open(os.path.join(DATA, "episode.json")))
st = b.steel


def our_structure(geo):
    """Rebuild an engine Structure whose node order matches geo (geo nodes already in Structure order)."""
    nodes = np.array(geo["nodes"])
    bars = np.array(geo["bars"])
    # geo comes from Structure.to_json-like export: kept-node order; fixed brief nodes are the deck nodes at y = 0
    full = []
    remap = {}
    for i, p in enumerate(nodes):
        d = np.hypot(*(b.fixed - p).T)
        j = int(np.argmin(d))
        if d[j] < 1e-6:
            remap[i] = j
    extra = [i for i in range(len(nodes)) if i not in remap]
    allnodes = np.vstack([b.fixed, nodes[extra]]) if extra else b.fixed.copy()
    for k, i in enumerate(extra):
        remap[i] = b.nf + k
    S = E.Structure(b, allnodes, np.array([[remap[a], remap[c]] for a, c in bars]))
    # map geo bar order -> S bar order
    key = {tuple(sorted((remap[a], remap[c]))): k for k, (a, c) in enumerate(bars)}
    order = [key[tuple(sorted((int(S.orig[a]), int(S.orig[c]))))] for a, c in S.bars]
    A = np.array(geo["A_m2"])[order]
    return S, A


def pynite_model(S, A, F2, rigid):
    m = FEModel3D()
    m.add_material("S355", st.E, st.E / 2.6, 0.3, st.rho)
    for i, (x, y) in enumerate(S.nodes):
        m.add_node(f"N{i}", float(x), float(y), 0.0)
    for k, (a, c) in enumerate(S.bars):
        I = st.kI * A[k] ** 2
        m.add_section(f"s{k}", A=float(A[k]), Iy=float(I), Iz=float(I), J=float(2 * I))
        m.add_member(f"M{k}", f"N{a}", f"N{c}", "S355", f"s{k}")
        if not rigid:
            m.def_releases(f"M{k}", False, False, False, True, True, True, False, False, False, False, True, True)
    fixed = set(int(d) for d in S.fixed_dofs)
    for i in range(S.n):
        fx, fy = (2 * i) in fixed, (2 * i + 1) in fixed
        if rigid:
            m.def_support(f"N{i}", fx, fy, True, True, True, False)
        else:
            m.def_support(f"N{i}", fx, fy, True, True, True, True)
    for i in range(S.n):
        if abs(F2[2 * i]) > 0:
            m.add_node_load(f"N{i}", "FX", float(F2[2 * i]))
        if abs(F2[2 * i + 1]) > 0:
            m.add_node_load(f"N{i}", "FY", float(F2[2 * i + 1]))
    m.analyze_linear(check_statics=False)
    N = np.array([-m.members[f"M{k}"].axial(0.5 * S.L[k]) for k in range(S.m)])
    Mmax = np.array([max(abs(m.members[f"M{k}"].max_moment("Mz")), abs(m.members[f"M{k}"].min_moment("Mz")))
                     for k in range(S.m)])
    return N, Mmax


def util_rigid(S, A, N, Mmax):
    Nt, Nc = st.caps(A, S.L)
    D = st.diameter(A)
    I = st.kI * A ** 2
    W = I / (D / 2)
    ax = np.where(N >= 0, N / Nt, -N / Nc)
    return ax + Mmax / (st.fy * W)


out = {"pinned": [], "rigid": []}
results = {}
for label, gkey in (("evolved", "evolved_equal_steel"), ("pratt", "pratt")):
    S, A = our_structure(ep["geometry"][gkey])
    for c in b.cases:
        Fd, Fl = S.case_loads(c)
        sw = S.selfweight(A)
        F2 = b.gam_d * Fd + b.gam_l * Fl + b.gam_s * sw
        Nours, _ = S.forces(A, F2)
        Np, _ = pynite_model(S, A, F2, rigid=False)
        big = np.abs(Nours) > 0.02 * np.abs(Nours).max()
        dev = float(np.max(np.abs(Np[big] - Nours[big]) / np.abs(Nours[big])))
        out["pinned"].append({"design": label, "case": c["name"], "max_rel_diff": dev,
                              "max_util_ours": float(S.util(A, Nours[:, None]).max()),
                              "max_util_pynite": float(S.util(A, Np[:, None]).max())})
        # rigid joints: first-failure load factor with bending, ramping the live load (dead + self at 1.0)
        Fd1 = Fd + sw
        Nd, Md = pynite_model(S, A, Fd1, rigid=True)
        Nl, Ml = pynite_model(S, A, Fl, rigid=True)
        lam_lo, lam_hi = 0.0, 10.0
        for _ in range(40):
            lam = 0.5 * (lam_lo + lam_hi)
            Nn = Nd + lam * Nl
            # moments add with sign along the member; use a conservative sum of magnitudes
            Mm = Md + lam * Ml
            if util_rigid(S, A, Nn, Mm).max() > 1.0:
                lam_hi = lam
            else:
                lam_lo = lam
        results[(label, c["name"])] = lam_lo
        out["rigid"].append({"design": label, "case": c["name"], "first_fail_lambda_rigid_bending": lam_lo})
        print(f"{label:8s} {c['name']:20s} pinned: max force diff {100 * dev:.3f}%   rigid+bending first failure x{lam_lo:.3f}")
for c in b.cases:
    e, p = results[("evolved", c["name"])], results[("pratt", c["name"])]
    ours = next(r for r in ep["cases"] if r["case"] == c["name"])
    out.setdefault("compare", []).append({"case": c["name"], "rigid_ratio": e / p, "pinned_ratio": ours["ratio_first"]})
    print(f"{c['name']:20s} equal steel, evolved/Pratt: pinned (ours) x{ours['ratio_first']:.2f}  rigid+bending (PyNite) x{e / p:.2f}")
json.dump(out, open(os.path.join(DATA, "pynite_check.json"), "w"), indent=1)
print("wrote pynite_check.json")

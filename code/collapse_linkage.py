"""Collapse motion after the first failure, from the solver's own model: the pin-jointed truss minus nothing, but with
the failed member given a mid joint (its plastic hinge, with a small bow), moving as a linkage under gravity.
Rigid bars = distance constraints (position-based dynamics), pinned left support, roller right support, the
quarry gorge as the ground. Masses per truss plane: members + joints + deck dead load + the test blocks on the
loaded half. Everything here happens after the solver's first failure, so it is labelled visualisation.
Writes ../data/collapse_<design>.json: per frame (30 fps) the 2D position of every node (+ the hinge node).
Usage: py -I collapse_linkage.py"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ep = json.load(open(os.path.join(DATA, "episode.json")))
G = 9.81
FPS = 30
SUB = 60
ITERS = 25
BLOCK_T = 3.456


def ground_z(x):
    """Quarry gorge profile (world x, metres): abutment shelves at -0.6, walls, floor at -16."""
    x = np.asarray(x, float)
    z = np.full_like(x, -16.0)
    left = x < 3.5
    right = x > 36.5
    z = np.where(left, -16.0 * np.clip((x + 1.0) / 4.5, 0, 1), z)
    z = np.where(right, -16.0 * np.clip((41.0 - x) / 4.5, 0, 1), z)
    z = np.where(x <= 0.6, np.maximum(z, -0.6), z)            # abutment top (bearing shelf)
    z = np.where(x >= 39.4, np.maximum(z, -0.6), z)
    return z


def simulate(geo_key, case_key, seconds=6.0, kink=0.06):
    g = ep["geometry"][geo_key]
    nodes = np.array(g["nodes"], float)
    bars = [tuple(b) for b in g["bars"]]
    A = np.array(g["A_m2"])
    case = next(c for c in ep["cases"] if c["case"] == "traffic left half")
    fail = case[case_key]["events"][0]["bar"]
    lam = case[case_key]["first_lam"]
    n = len(nodes)
    # masses per truss plane (kg)
    m = np.zeros(n)
    Lb = np.array([np.linalg.norm(nodes[b] - nodes[a]) for a, b in bars])
    for (a, b), Ai, Li in zip(bars, A, Lb):
        m[a] += 7850 * Ai * Li / 2
        m[b] += 7850 * Ai * Li / 2
    m += ep["joint_mass_t"] * 1000
    deck = [i for i in range(n) if abs(nodes[i, 1]) < 1e-6]
    for i in deck:
        x = nodes[i, 0]
        trib = 2.0 if x in (0.0, 40.0) else 4.0
        m[i] += 30e3 * trib / G                                  # deck dead load per truss
        if x < 20 - 1e-6:
            m[i] += lam * 40e3 * trib / G                         # blocks at the failure load (left half)
        elif abs(x - 20) < 1e-6:
            m[i] += lam * 40e3 * trib / G * 0.5
    # the plastic hinge: split the failed member at mid with a small in-plane bow
    a, b = bars[fail]
    pa, pb = nodes[a], nodes[b]
    u = (pb - pa) / np.linalg.norm(pb - pa)
    nrm = np.array([-u[1], u[0]])
    # bow towards the side the member would buckle to (away from the truss centre line, a coin we fix)
    mid = (pa + pb) / 2 + nrm * kink * np.linalg.norm(pb - pa) / 2
    nodes = np.vstack([nodes, mid])
    m = np.append(m, 7850 * A[fail] * Lb[fail] * 0.2)
    k = n
    cons = [(i, j) for idx, (i, j) in enumerate(bars) if idx != fail] + [(a, k), (k, b)]
    L0 = np.array([np.linalg.norm(nodes[j] - nodes[i]) for i, j in cons])
    ci = np.array([c[0] for c in cons])
    cj = np.array([c[1] for c in cons])
    w = 1.0 / m
    # member capacities (the solver's own checks): tension yield, EC3 buckling; the hinged member's halves keep tension only
    import sys
    sys.path.insert(0, HERE)
    from ndt_engine import Steel
    st = Steel(Dt=25.0, A_min=1e-3)
    orig_idx = [idx for idx in range(len(bars)) if idx != fail] + [fail, fail]
    Ac = np.array([A[i] for i in orig_idx])
    Lc = np.array([Lb[i] for i in orig_idx])
    Nt, Nc = st.caps(Ac, Lc)
    Nc[-2:] = 0.0                                              # the buckled member's halves: no compression left
    alive = np.ones(len(cons), bool)
    broken = []
    sup_l = int(np.argmin(np.hypot(nodes[:, 0] - 0, nodes[:, 1])))
    sup_r = int(np.argmin(np.hypot(nodes[:, 0] - 40, nodes[:, 1])))
    x = nodes.copy()
    xp = x.copy()
    dt = 1.0 / FPS / SUB
    frames = [x.copy()]
    pinned_r = True
    for f in range(int(seconds * FPS)):
        for s in range(SUB):
            v = (x - xp) * 0.9995
            xp = x.copy()
            x = x + v
            x[:, 1] -= G * dt * dt
            for it in range(ITERS):
                d = x[cj] - x[ci]
                L = np.linalg.norm(d, axis=1)
                if it == 0 and f > 2:
                    # member force from the constraint: stretch error * effective mass / dt^2 (tension +)
                    meff = 1.0 / (w[ci] + w[cj])
                    F = (L - L0) * meff / dt ** 2
                    over = alive & (((F > 0) & (F > Nt)) | ((F < 0) & (-F > Nc + 1e-9) & (Nc > 0)))
                    for q in np.flatnonzero(over):
                        alive[q] = False
                        broken.append({"frame": f, "bar": int(orig_idx[q]), "mode": "yield" if F[q] > 0 else "buckle"})
                corr = (((L - L0) / np.maximum(L, 1e-9)) * alive)[:, None] * d
                # the hinged halves can only pull: a shortened half is slack
                slack = np.zeros(len(cons), bool)
                slack[-2:] = L[-2:] < L0[-2:]
                corr[slack] = 0.0
                wi, wj = w[ci][:, None], w[cj][:, None]
                s_ = wi + wj
                dxi = corr * (wi / s_)
                dxj = -corr * (wj / s_)
                np.add.at(x, ci, dxi * 0.5)
                np.add.at(x, cj, dxj * 0.5)
                # supports: left pin, right roller on its shelf (until it is dragged off the edge)
                x[sup_l] = nodes[sup_l]
                if pinned_r:
                    x[sup_r, 1] = nodes[sup_r, 1]
                    if x[sup_r, 0] < 39.4:
                        pinned_r = False
                # ground
                gz = ground_z(x[:, 0])
                below = x[:, 1] < gz
                if below.any():
                    x[below, 1] = gz[below]
                    xp[below, 0] = x[below, 0] - (x[below, 0] - xp[below, 0]) * 0.3   # friction
                    xp[below, 1] = x[below, 1]
        frames.append(x.copy())
    return {"design": geo_key, "fail_bar": int(fail), "hinge_node": int(k), "bars": [list(map(int, c)) for c in cons],
            "broken": broken,
            "orig_bars": [list(map(int, b)) for b in bars], "fps": FPS,
            "frames": [np.round(fr, 4).tolist() for fr in frames]}


if __name__ == "__main__":
    for key, ck, name in (("pratt", "pratt", "pratt"), ("evolved_equal_steel", "evolved", "evolved")):
        r = simulate(key, ck)
        fr = np.array(r["frames"])
        drop = fr[:, :, 1].min(axis=1)
        print(name, "lowest joint z at 0.5/1/2/3/6 s:", [round(float(drop[int(t * FPS)]), 2) for t in (0.5, 1, 2, 3, 6)],
              "| members that failed during the motion:", [(b["frame"], b["bar"], b["mode"]) for b in r["broken"]][:12])
        json.dump(r, open(os.path.join(DATA, f"collapse_{name}.json"), "w"), separators=(",", ":"))

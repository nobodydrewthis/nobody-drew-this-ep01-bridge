"""Break-test motion for Blender: an explicit large-displacement simulation of each truss plane (XPBD with real
member stiffness EA/L, pinned joints). Members are elastic-plastic: tension capped at the yield force (plastic
stretch), compression capped at the EC3 buckling resistance; once a member buckles its capacity drops to 30%
(post-buckling softening). The live load ramps slowly (quasi-static) from 0 to the solver's first-failure load, so
the first member to give up should be the solver's, at the solver's load; the test blocks stop there (no more are
added) and the motion after that is the visualisation. The gorge is the ground.
Writes ../data/collapse_<design>.json (30 fps): node positions per frame, member states, failure events.
Usage: py -I collapse_xpbd.py"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ndt_engine import Steel
from collapse_linkage import ground_z

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ep = json.load(open(os.path.join(DATA, "episode.json")))
G = 9.81
FPS = 30
SUB = 90
ITERS = 6
st = Steel(Dt=25.0, A_min=1e-3)


def run(geo_key, case_key, ramp_s=4.0, after_s=7.0, lam_over=1.0):
    g = ep["geometry"][geo_key]
    nodes = np.array(g["nodes"], float)
    bars = np.array(g["bars"], int)
    A = np.array(g["A_m2"])
    case = next(c for c in ep["cases"] if c["case"] == "traffic left half")
    lam_fail = case[case_key]["first_lam"]
    fail_bar = case[case_key]["events"][0]["bar"]
    n, m = len(nodes), len(bars)
    L0 = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    Nt, Nc = st.caps(A, L0)
    Ncap = Nc.copy()
    alpha = L0 / (st.E * A)                     # compliance
    # masses (per truss plane) and loads
    mass = np.zeros(n)
    np.add.at(mass, bars[:, 0], st.rho * A * L0 / 2)
    np.add.at(mass, bars[:, 1], st.rho * A * L0 / 2)
    mass += ep["joint_mass_t"] * 1000
    dead = np.zeros(n)
    live = np.zeros(n)
    for i, (x, y) in enumerate(nodes):
        if abs(y) < 1e-6:
            trib = 2.0 if x in (0.0, 40.0) else 4.0
            dead[i] += 30e3 * trib
            live[i] += 40e3 * trib * (1.0 if x < 20 - 1e-6 else 0.5 if abs(x - 20) < 1e-6 else 0.0)
    # the deck and the blocks are carried as mass on the deck nodes (they fall with the deck)
    mass_dead = mass + dead / G
    i0 = int(np.argmin(np.hypot(nodes[:, 0], nodes[:, 1])))
    i1 = int(np.argmin(np.hypot(nodes[:, 0] - 40, nodes[:, 1])))
    x = nodes.copy()
    v = np.zeros_like(x)
    lamb = np.zeros(m)
    state = np.zeros(m, int)                    # 0 elastic, 1 buckled, 2 yielded (snapped)
    rest = L0.copy()
    events = []
    frames = [x.copy()]
    states = [state.copy()]
    dt = 1.0 / FPS / SUB
    total = int((ramp_s + after_s) * FPS)
    lam_hold = None
    roller_on = True
    for f in range(total):
        t = (f + 1) / FPS
        lam = lam_fail * lam_over * min(1.0, t / ramp_s) if lam_hold is None else lam_hold
        if lam_hold is None and t >= ramp_s:
            # the solver's first failure, exactly where and when the solver says (the gate), then the sim takes over
            d = x[bars[:, 1]] - x[bars[:, 0]]
            Fnow = (np.linalg.norm(d, axis=1) - rest) / alpha
            util = np.where(Fnow >= 0, Fnow / Nt, -Fnow / Nc)
            state[fail_bar] = 1
            Ncap[fail_bar] = 0.3 * Nc[fail_bar]
            lam_hold = lam
            events.append({"t": t, "frame": f + 1, "bar": int(fail_bar), "mode": "buckles (solver)", "lam": lam,
                           "sim_util_of_this_bar": float(util[fail_bar]), "sim_max_util": float(util.max()),
                           "sim_max_util_bar": int(util.argmax())})
        ml = mass_dead + lam * live / G
        w = 1.0 / ml
        w[i0] = 0.0
        for s in range(SUB):
            # gravity, damping (critical-ish on the slow ramp, light after the failure)
            damp = 0.995 if lam_hold is None else 0.9995
            v = v * damp
            v[:, 1] -= G * dt
            xp = x.copy()
            x = x + v * dt
            lamb[:] = 0.0
            on = state != 2
            for _ in range(ITERS):
                d = x[bars[:, 1]] - x[bars[:, 0]]
                L = np.linalg.norm(d, axis=1)
                nvec = d / np.maximum(L, 1e-9)[:, None]
                C = L - rest
                at = alpha / dt ** 2
                wi, wj = w[bars[:, 0]], w[bars[:, 1]]
                dl = (-C - at * lamb) / (wi + wj + at)
                dl = np.where(on, dl, 0.0)
                lamb += dl
                np.add.at(x, bars[:, 0], -(wi * dl)[:, None] * nvec)
                np.add.at(x, bars[:, 1], (wj * dl)[:, None] * nvec)
                x[i0] = nodes[i0]
                if roller_on:
                    x[i1, 1] = nodes[i1, 1]
                    if x[i1, 0] < 39.4:
                        roller_on = False
            # member forces (tension +) from each member's elastic stretch (Hooke), then plasticity
            d = x[bars[:, 1]] - x[bars[:, 0]]
            F = (np.linalg.norm(d, axis=1) - rest) / alpha
            ten = on & (F > Nt)
            com = on & (F < -Ncap)
            if ten.any():
                rest[ten] += (F[ten] - Nt[ten]) * alpha[ten]
                for k in np.flatnonzero(ten & (rest > L0 * 1.06)):      # 6% plastic strain: it tears
                    if state[k] != 2:
                        state[k] = 2
                        events.append({"t": t, "frame": f + 1, "bar": int(k), "mode": "tears", "lam": lam})
            if com.any():
                rest[com] -= (-F[com] - Ncap[com]) * alpha[com]
                for k in np.flatnonzero(com & (state == 0)):
                    state[k] = 1
                    Ncap[k] = 0.3 * Nc[k]
                    events.append({"t": t, "frame": f + 1, "bar": int(k), "mode": "buckles", "lam": lam})
                    if lam_hold is None:
                        lam_hold = lam                                    # the blocks stop: hold the load here
            # ground
            gz = ground_z(x[:, 0])
            below = x[:, 1] < gz
            if below.any():
                x[below, 1] = gz[below]
                v[below] *= 0.2
            v = (x - xp) / dt
        frames.append(x.copy())
        states.append(state.copy())
    first = events[0] if events else None
    return {"design": geo_key, "solver_first_bar": int(fail_bar), "solver_first_lam": lam_fail, "first_event": first,
            "events": events, "fps": FPS, "ramp_s": ramp_s, "bars": bars.tolist(),
            "frames": [np.round(fr, 4).tolist() for fr in frames], "states": [s.tolist() for s in states]}


if __name__ == "__main__":
    for key, ck, name in (("pratt", "pratt", "pratt"), ("evolved_equal_steel", "evolved", "evolved")):
        r = run(key, ck)
        fr = np.array(r["frames"])
        e0 = r["first_event"]
        print(f"{name}: solver says bar {r['solver_first_bar']} at x{r['solver_first_lam']:.3f}; "
              f"simulation: first {e0['mode'] if e0 else '-'} bar {e0['bar'] if e0 else '-'} at x{e0['lam'] if e0 else 0:.3f}, t={e0['t'] if e0 else 0:.2f}s")
        print("   later events:", [(round(e['t'], 2), e['bar'], e['mode']) for e in r["events"][1:10]])
        drop = fr[:, :, 1].min(axis=1)
        print("   lowest joint z at 4.5/5/6/8/11 s:", [round(float(drop[min(len(drop) - 1, int(t * FPS))]), 2) for t in (4.5, 5, 6, 8, 11)])
        json.dump(r, open(os.path.join(DATA, f"collapse_{name}.json"), "w"), separators=(",", ":"))

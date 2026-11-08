"""Candidate briefs for Nobody Drew This, episode 1. Fixed before any run (see PROTOCOL.md).
Units: metres, newtons. Loads are per plane truss. gamma: 1.35 dead / self-weight, 1.5 live unless stated."""
import numpy as np
from ndt_engine import Brief, Steel, G

STEEL = dict(E=210e9, fy=355e6, rho=7850.0, Dt=25.0, alpha=0.21, lam_c=200.0, name="S355 steel tube, D/t 25")


def zeros(nf):
    return np.zeros((nf, 2))


def rect(x0, y0, x1, y1):
    return [(x0, y0, x1, y1)]


# ------------------------------------------------------------------------------------------- 1 road bridge 40 m
def bridge40():
    span, p = 40.0, 4.0
    xs = np.arange(0, span + 1e-9, p)                      # 11 deck nodes
    fixed = [(x, 0.0) for x in xs]
    nf = len(fixed)
    sup = {0: (True, True), nf - 1: (False, True)}
    trib = np.where((xs == 0) | (xs == span), p / 2, p)
    dead = 30e3 * trib                                     # deck, surfacing, cross girders: 30 kN/m per truss
    live = 40e3 * trib                                     # traffic: 40 kN/m per truss
    def case(name, wmask):
        Fd, Fl = zeros(nf), zeros(nf)
        Fd[:, 1] = -dead
        Fl[:, 1] = -live * wmask
        return {"name": name, "Fd": Fd, "Fl": Fl}
    half = span / 2
    left = np.where(xs < half - 1e-9, 1.0, np.where(np.abs(xs - half) < 1e-9, 0.5, 0.0))
    cases = [case("traffic full", np.ones(nf)), case("traffic left half", left), case("traffic right half", left[::-1])]
    # unseen: a 180 t abnormal load (transformer haul), 900 kN per truss on two axle groups at 12 m and 16 m
    Fd, Fl = zeros(nf), zeros(nf)
    Fd[:, 1] = -dead
    Fl[3, 1] = Fl[4, 1] = -450e3
    unseen = [{"name": "abnormal load at 12-16 m", "Fd": Fd, "Fl": Fl}]
    # textbook: Pratt truss, depth 5 m (span/8), 4 m panels, inclined end posts
    h = 5.0
    top = [(x, h) for x in xs[1:-1]]                       # 9 top nodes, ids nf..nf+8
    nodes = fixed + top
    T = nf
    bars = [(i, i + 1) for i in range(nf - 1)]
    bars += [(T + i, T + i + 1) for i in range(len(top) - 1)]
    bars += [(0, T), (nf - 1, T + len(top) - 1)]
    bars += [(i, T + i - 1) for i in range(1, nf - 1)]     # verticals
    mid = nf // 2                                          # deck node at 20 m
    bars += [(T + i - 1, i + 1) for i in range(1, mid)]    # left diagonals slope down to the centre
    bars += [(T + i - 1, i - 1) for i in range(mid + 1, nf - 1)]
    return Brief("bridge40", "40 m road bridge vs the Pratt truss", fixed, sup, cases, rect(0, 0, span, 8.0),
                 Steel(**STEEL, A_min=1.0e-3), L_min=1.5, L_max=12.0, clear=0.35, m_joint=0.20,
                 k_free=(4, 12), sym_x=span / 2, unseen=unseen,
                 textbook={"name": "Pratt truss (1844), 5 m deep", "nodes": nodes, "bars": bars},
                 notes="Through truss: structure above the deck only. Pin + roller supports.")


# ------------------------------------------------------------------------------------------- 2 railway bridge 30 m
def rail30():
    span, p = 30.0, 3.0
    xs = np.arange(0, span + 1e-9, p)
    fixed = [(x, 0.0) for x in xs]
    nf = len(fixed)
    sup = {0: (True, True), nf - 1: (False, True)}
    trib = np.where((xs == 0) | (xs == span), p / 2, p)
    dead = 45e3 * trib                                     # ballast, sleepers, deck
    udl = 40e3 * trib                                      # train UDL per truss
    cases = []
    for k in range(1, nf - 2):                             # locomotive: 2 x 250 kN per truss on nodes k, k+1
        Fd, Fl = zeros(nf), zeros(nf)
        Fd[:, 1] = -dead
        Fl[:, 1] = -udl
        Fl[k, 1] -= 250e3
        Fl[k + 1, 1] -= 250e3
        cases.append({"name": f"locomotive at {xs[k]:.0f}-{xs[k + 1]:.0f} m", "Fd": Fd, "Fl": Fl, "gl": 1.45})
    Fd, Fl = zeros(nf), zeros(nf)
    Fd[:, 1] = -dead
    Fl[2, 1] = -900e3
    unseen = [{"name": "derailed axle load at 6 m", "Fd": Fd, "Fl": Fl}]
    h = 4.5
    top = [(x, h) for x in xs[1:-1]]
    nodes = fixed + top
    T = nf
    bars = [(i, i + 1) for i in range(nf - 1)] + [(T + i, T + i + 1) for i in range(len(top) - 1)]
    bars += [(0, T), (nf - 1, T + len(top) - 1)]
    bars += [(i, T + i - 1) for i in range(1, nf - 1)]     # verticals
    # Warren diagonals with verticals: alternate direction panel by panel
    for i in range(1, nf - 2):
        bars.append((i, T + i) if i % 2 == 1 else (i + 1, T + i - 1))
    return Brief("rail30", "30 m railway bridge vs the Warren truss", fixed, sup, cases, rect(0, 0, span, 7.0),
                 Steel(**STEEL, A_min=1.0e-3), L_min=1.2, L_max=10.0, clear=0.3, m_joint=0.15,
                 k_free=(4, 10), sym_x=span / 2, unseen=unseen,
                 textbook={"name": "Warren truss with verticals, 4.5 m deep", "nodes": nodes, "bars": bars},
                 notes="Moving locomotive load cases (8). Structure above the track only.")


# ------------------------------------------------------------------------------------------- 3 hammerhead crane jib 50 m
def jib50():
    R, p = 50.0, 2.5
    tower = [(0.0, float(y)) for y in range(0, 10)]        # tower face nodes 0..9 m (pins)
    rail = [(x, 0.0) for x in np.arange(p, R + 1e-9, p)]   # 20 rail nodes
    fixed = tower + rail
    nf = len(fixed)
    sup = {i: (True, True) for i in range(len(tower))}
    nt = len(tower)
    rx = np.array([x for x, _ in rail])
    dead = 0.15e3 * G * np.where(rx == R, p / 2, p)       # rail, walkway, trolley track: 0.15 t/m
    cases = []
    for k, r in enumerate(rx):
        payload = min(8.0, 120.0 / r)                     # load chart: 8 t to 15 m, then 120 t.m
        Fd, Fl = zeros(nf), zeros(nf)
        Fd[nt:, 1] = -dead
        Fl[nt + k, 1] = -(payload + 0.6) * 1e3 * G * 1.15  # + trolley and hook 0.6 t, hoisting factor 1.15
        cases.append({"name": f"{payload:.1f} t at {r:.1f} m", "Fd": Fd, "Fl": Fl})
    Fd, Fl = zeros(nf), zeros(nf)
    Fd[nt:, 1] = -dead
    Fl[nf - 1, 1] = -(8.0 + 0.6) * 1e3 * G
    unseen = [{"name": "8 t at the tip (chart allows 2.4 t)", "Fd": Fd, "Fl": Fl}]
    # textbook: Warren lattice jib 2.0 m deep + two pendant ties from the 9 m apex
    d = 2.0
    topx = [p / 2 + p * i for i in range(20)]              # 1.25 ... 48.75
    nodes = fixed + [(x, d) for x in topx]
    T = nf
    bars = [(0, nt)] + [(nt + i, nt + i + 1) for i in range(19)]           # bottom chord (rail)
    bars += [(2, T)] + [(T + i, T + i + 1) for i in range(19)]             # top chord from the tower at 2 m
    bars += [(0, T)]
    for i in range(20):
        bars.append((T + i, nt + i))                                        # down to the next rail node
        if i < 19:
            bars.append((nt + i, T + i + 1))
    bars += [(9, T + 7), (9, T + 14)]                                       # pendants to 18.75 m and 36.25 m
    return Brief("jib50", "50 m hammerhead crane jib vs the lattice jib", fixed, sup, cases, rect(0, 0, R, 9.0),
                 Steel(**{**STEEL, "name": "S355 tube, tension members may be pendant rods"}, A_min=4.0e-4,
                       lam_t=None),
                 L_min=1.2, L_max=40.0, clear=0.2, m_joint=0.04, k_free=(8, 24), sym_x=None, unseen=unseen,
                 textbook={"name": "Warren lattice jib, 2 m deep, two pendants from a 9 m apex", "nodes": nodes,
                           "bars": bars},
                 notes="20 trolley positions with the load chart. Tower treated as a rigid support face 0-9 m.")


# ------------------------------------------------------------------------------------------- 4 flat-top crane jib 50 m
def jib50flat():
    R, p = 50.0, 2.5
    tower = [(0.0, y) for y in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)]
    rail = [(x, 0.0) for x in np.arange(p, R + 1e-9, p)]
    fixed = tower + rail
    nf = len(fixed)
    nt = len(tower)
    sup = {i: (True, True) for i in range(nt)}
    rx = np.array([x for x, _ in rail])
    dead = 0.15e3 * G * np.where(rx == R, p / 2, p)
    cases = []
    for k, r in enumerate(rx):
        payload = min(6.0, 90.0 / r)
        Fd, Fl = zeros(nf), zeros(nf)
        Fd[nt:, 1] = -dead
        Fl[nt + k, 1] = -(payload + 0.5) * 1e3 * G * 1.15
        cases.append({"name": f"{payload:.1f} t at {r:.1f} m", "Fd": Fd, "Fl": Fl})
    Fd, Fl = zeros(nf), zeros(nf)
    Fd[nt:, 1] = -dead
    Fl[nf - 1, 1] = -(6.0 + 0.5) * 1e3 * G
    unseen = [{"name": "6 t at the tip (chart allows 1.8 t)", "Fd": Fd, "Fl": Fl}]
    d = 2.0
    topx = [p / 2 + p * i for i in range(20)]
    nodes = fixed + [(x, d) for x in topx]
    T = nf
    bars = [(0, nt)] + [(nt + i, nt + i + 1) for i in range(19)]
    bars += [(4, T)] + [(T + i, T + i + 1) for i in range(19)]
    bars += [(0, T)]
    for i in range(20):
        bars.append((T + i, nt + i))
        if i < 19:
            bars.append((nt + i, T + i + 1))
    return Brief("jib50flat", "50 m flat-top crane jib vs the lattice jib", fixed, sup, cases, rect(0, 0, R, 4.0),
                 Steel(**STEEL, A_min=4.0e-4), L_min=1.0, L_max=12.0, clear=0.15, m_joint=0.04,
                 k_free=(8, 24), sym_x=None, unseen=unseen,
                 textbook={"name": "Warren lattice jib, 2 m deep (flat-top)", "nodes": nodes, "bars": bars},
                 notes="No apex: the jib must stay within 4 m of height. Tower connection face 0-2.5 m.")


# ------------------------------------------------------------------------------------------- 5 cantilever (Michell)
def cantilever20():
    wall = [(0.0, float(y)) for y in range(-5, 6)]          # 11 wall nodes
    fixed = wall + [(20.0, 0.0)]
    nf = len(fixed)
    sup = {i: (True, True) for i in range(len(wall))}
    Fd, Fl = zeros(nf), zeros(nf)
    Fl[nf - 1, 1] = -600e3
    cases = [{"name": "600 kN at the tip", "Fd": Fd, "Fl": Fl}]
    Fd2, Fl2 = zeros(nf), zeros(nf)
    Fl2[nf - 1] = (300e3, -520e3)
    unseen = [{"name": "tip load tilted 30 degrees outward", "Fd": Fd2, "Fl": Fl2}]
    xs = [4.0, 8.0, 12.0, 16.0]
    top = [(x, 4 * (1 - x / 20)) for x in xs]
    bot = [(x, -4 * (1 - x / 20)) for x in xs]
    nodes = fixed + top + bot
    T, B = nf, nf + 4
    tip = nf - 1
    wt, wb = 9, 1                                           # wall nodes at +4 and -4
    bars = [(wt, T), (T, T + 1), (T + 1, T + 2), (T + 2, T + 3), (T + 3, tip)]
    bars += [(wb, B), (B, B + 1), (B + 1, B + 2), (B + 2, B + 3), (B + 3, tip)]
    bars += [(T + i, B + i) for i in range(4)]             # verticals
    bars += [(wt, B)] + [(T + i, B + i + 1) for i in range(3)]   # diagonals
    return Brief("cantilever20", "20 m cantilever vs the tapered truss (Michell's problem)", fixed, sup, cases,
                 rect(0, -5, 20, 5), Steel(**STEEL, A_min=1.0e-3), L_min=1.0, L_max=12.0, clear=0.3, m_joint=0.10,
                 k_free=(6, 20), sym_x=None, unseen=unseen,
                 textbook={"name": "tapered N-truss cantilever, 8 m deep at the wall", "nodes": nodes, "bars": bars},
                 notes="Single tip load (Michell 1904). Wall face -5..5 m.", gam_l=1.5)


# ------------------------------------------------------------------------------------------- 6 canopy on one foundation (tree)
def canopy24():
    roofx = [float(x) for x in range(-11, 12, 2)]          # 12 roof nodes at y = 10
    feet = [(x, 0.0) for x in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)]
    fixed = [(x, 10.0) for x in roofx] + feet
    nf = len(fixed)
    nr = len(roofx)
    sup = {nr + i: (True, True) for i in range(len(feet))}
    trib = 2.0 * 8.0                                        # 2 m x 8 m bay per roof node
    dead = 0.6e3 * trib
    snow = 0.8e3 * trib
    rx = np.array(roofx)
    def mk(name, snow_mask=None, uplift=None, side=0.0, gd=None):
        Fd, Fl = zeros(nf), zeros(nf)
        Fd[:nr, 1] = -dead
        if snow_mask is not None:
            Fl[:nr, 1] = -snow * snow_mask
        if uplift is not None:
            Fl[:nr, 1] = uplift * trib
            Fl[:nr, 0] = side * trib
        c = {"name": name, "Fd": Fd, "Fl": Fl}
        if gd is not None:
            c["gd"] = gd
        return c
    left = (rx < 0).astype(float)
    up = np.where(rx < 0, 0.7e3, 0.35e3)
    cases = [mk("snow full", np.ones(nr)), mk("snow left side", left), mk("snow right side", left[::-1]),
             mk("wind from the left", uplift=up, side=0.15e3, gd=1.0),
             mk("wind from the right", uplift=up[::-1], side=-0.15e3, gd=1.0)]
    Fd, Fl = zeros(nf), zeros(nf)
    Fd[:nr, 1] = -dead
    Fl[0, 1] = -60e3
    Fl[1, 1] = -60e3
    unseen = [{"name": "12 t of drifted snow at the left edge", "Fd": Fd, "Fl": Fl}]
    domain = [(-12.0, 4.5, 12.0, 10.0), (-1.5, 0.0, 1.5, 4.5)]          # canopy zone + trunk zone
    # textbook: umbrella canopy = lattice mast (chords at x = +-1) + two tapered cantilever trusses
    nodes = list(fixed)
    def add(p):
        nodes.append(p)
        return len(nodes) - 1
    iL = [roofx.index(-1.0), roofx.index(1.0)]
    fL, fR = nr + 1, nr + 5                                 # feet at x = -1, +1
    mastL = [fL] + [add((-1.0, y)) for y in (2.5, 5.0, 7.5)] + [iL[0]]
    mastR = [fR] + [add((1.0, y)) for y in (2.5, 5.0, 7.5)] + [iL[1]]
    bars = [(mastL[i], mastL[i + 1]) for i in range(4)] + [(mastR[i], mastR[i + 1]) for i in range(4)]
    bars += [(mastL[i], mastR[i]) for i in range(1, 5)]                     # rungs
    bars += [(mastL[i], mastR[i + 1]) if i % 2 == 0 else (mastR[i], mastL[i + 1]) for i in range(4)]
    bars += [(i, i + 1) for i in range(nr - 1)]                             # roof top chord
    # cantilever bottom chords: from the mast at 7.5 m to the roof edge at 9.6 m
    for side, mast in ((-1, mastL), (1, mastR)):
        xs = [side * x for x in (3.0, 5.0, 7.0, 9.0, 11.0)]
        prev = mast[3]
        for k, x in enumerate(xs):
            y = 7.5 + (abs(x) - 1.0) / 10.0 * 2.1
            ri = roofx.index(x)
            if abs(x) < 11.0:
                b = add((x, y))
                bars += [(prev, b), (b, ri)]
                # Pratt-style diagonal toward the mast
                rprev = roofx.index(x - side * 2.0)
                bars.append((rprev, b))
                prev = b
            else:
                bars.append((prev, ri))
    return Brief("canopy24", "24 m canopy on one foundation vs the umbrella canopy", fixed, sup, cases, domain,
                 Steel(**STEEL, A_min=3.0e-4), L_min=0.8, L_max=8.0, clear=0.25, m_joint=0.03,
                 k_free=(5, 14), sym_x=0.0, unseen=unseen,
                 textbook={"name": "umbrella canopy: lattice mast + tapered cantilever trusses", "nodes": nodes,
                           "bars": bars},
                 notes="Clear height 4.5 m except a 3 m wide trunk zone. Snow patterns and wind uplift.")


# ------------------------------------------------------------------------------------------- 7 balsa competition bridge
def balsa30():
    fixed = [(0.0, 0.0), (0.15, 0.0), (0.30, 0.0)]
    nf = 3
    sup = {0: (True, True), 2: (False, True)}
    Fd, Fl = zeros(nf), zeros(nf)
    Fl[1, 1] = -20.0 * G                                    # 20 kg at the centre
    cases = [{"name": "20 kg at the centre", "Fd": Fd, "Fl": Fl}]
    Fd2, Fl2 = zeros(nf), zeros(nf)
    Fl2[1] = (0.0, 0.0)
    unseen = []
    # textbook: Pratt, 6 panels of 50 mm, 60 mm deep
    xs = [0.05, 0.10, 0.20, 0.25]
    nodes = list(fixed) + [(x, 0.0) for x in xs] + [(x, 0.06) for x in (0.05, 0.10, 0.15, 0.20, 0.25)]
    b0 = {0.0: 0, 0.05: 3, 0.10: 4, 0.15: 1, 0.20: 5, 0.25: 6, 0.30: 2}
    t0 = {0.05: 7, 0.10: 8, 0.15: 9, 0.20: 10, 0.25: 11}
    order = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    bars = [(b0[order[i]], b0[order[i + 1]]) for i in range(6)]
    bars += [(t0[0.05], t0[0.10]), (t0[0.10], t0[0.15]), (t0[0.15], t0[0.20]), (t0[0.20], t0[0.25])]
    bars += [(0, t0[0.05]), (2, t0[0.25])]
    bars += [(b0[x], t0[x]) for x in (0.05, 0.10, 0.15, 0.20, 0.25)]
    bars += [(t0[0.05], b0[0.10]), (t0[0.10], b0[0.15]), (t0[0.25], b0[0.20]), (t0[0.20], b0[0.15])]
    balsa = Steel(E=3.5e9, fy=20e6, fc=12e6, rho=160.0, alpha=0.2, plateau=0.3, section="square",
                  A_min=(1.6e-3) ** 2, lam_c=200.0, lam_t=None, name="balsa 160 kg/m3, square sticks")
    return Brief("balsa30", "30 cm balsa competition bridge vs the Pratt", fixed, sup, cases, rect(0, 0, 0.30, 0.10),
                 balsa, L_min=0.012, L_max=0.15, clear=0.003, m_joint=3e-8, k_free=(3, 10), sym_x=0.15,
                 unseen=unseen, gam_d=1.0, gam_l=1.0, gam_s=1.0,
                 textbook={"name": "Pratt truss, 60 mm deep", "nodes": nodes, "bars": bars},
                 notes="Competition style: one centre load, roller support, joints assumed stronger than sticks.")


ALL = {"bridge40": bridge40, "rail30": rail30, "jib50": jib50, "jib50flat": jib50flat,
       "cantilever20": cantilever20, "canopy24": canopy24, "balsa30": balsa30}

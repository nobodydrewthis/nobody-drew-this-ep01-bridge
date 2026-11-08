"""Nobody Drew This - structural engine (2D, pin-jointed, linear elastic) and evolutionary designer.

Honest checks (the same rules for the evolved design and the textbook design):
  * yield in tension and compression (fy),
  * member buckling with the Eurocode 3 column curve 'a' (imperfections included, not bare Euler),
  * slenderness limits L/r <= 200 (compression) and 300 (tension),
  * global stability: the stiffness matrix must be positive definite (no mechanisms),
  * pattern and moving load cases, factored loads, self-weight and joint weight included,
  * minimum and maximum member length, joint clearance, no crossing members.
Members are circular hollow steel sections with a fixed diameter/thickness ratio; areas are continuous.
Sizing is fully-stressed design (the classic engineering rule), applied identically to every design.
"""
import math
import json
import numpy as np

G = 9.81


# ----------------------------------------------------------------------------------------------- material
class Steel:
    """Member material + section. 'chs': circular hollow section with fixed D/t (steel, EC3 curve via alpha);
    'square': solid square stick (timber/balsa). Compression resistance = chi * A * fc with the EC3/EC5 form
    chi = 1/(phi + sqrt(phi^2 - lam^2)), phi = 0.5(1 + alpha(lam - plateau) + lam^2)."""

    def __init__(self, E=210e9, fy=355e6, rho=7850.0, Dt=25.0, alpha=0.21, A_min=1.0e-3,
                 lam_c=200.0, lam_t=300.0, name="S355 steel tube", fc=None, plateau=0.2, section="chs"):
        self.E, self.fy, self.rho, self.Dt, self.alpha = E, fy, rho, Dt, alpha
        self.fc = fy if fc is None else fc
        self.plateau, self.section = plateau, section
        self.A_min, self.lam_c, self.lam_t, self.name = A_min, lam_c, lam_t, name
        if section == "chs":
            q = 1.0 - 2.0 / Dt
            self.cA = math.pi / 4 * (1 - q * q)          # A = cA D^2
            self.cI = math.pi / 64 * (1 - q ** 4)        # I = cI D^4
        else:
            self.cA, self.cI = 1.0, 1.0 / 12.0           # A = b^2, I = b^4/12
        self.kI = self.cI / self.cA ** 2                 # I = kI A^2
        self.kr = math.sqrt(self.kI)                     # r = kr sqrt(A)
        self.clam = math.sqrt(self.fc / (math.pi ** 2 * E * self.kI))   # lambda_bar = clam * L / sqrt(A)
        lam = np.geomspace(1e-4, 30.0, 6000)
        h = self.chi(lam) / lam ** 2                     # strictly decreasing
        self._lam = lam
        self._x = -np.log(h)                             # increasing

    def to_json(self):
        return {k: getattr(self, k) for k in ("name", "E", "fy", "fc", "rho", "Dt", "alpha", "plateau", "section",
                                                 "A_min", "lam_c", "lam_t")}

    def chi(self, lam):
        phi = 0.5 * (1 + self.alpha * (lam - self.plateau) + lam ** 2)
        c = 1.0 / (phi + np.sqrt(np.maximum(phi ** 2 - lam ** 2, 0.0)))
        return np.where(lam <= self.plateau, 1.0, np.minimum(c, 1.0))

    def diameter(self, A):
        """Outer diameter (chs) or side (square), m."""
        return np.sqrt(A / self.cA)

    def caps(self, A, L):
        """Tension and compression resistance (N, positive numbers)."""
        lam = self.clam * L / np.sqrt(A)
        return A * self.fy, self.chi(lam) * A * self.fc

    def area_req(self, N, L):
        """Smallest area that carries signed force N (tension +) over length L, with all limits."""
        N = np.asarray(N, float)
        L = np.broadcast_to(L, N.shape)
        A = np.abs(N) / self.fy
        c = N < 0
        if np.any(c):
            q = -N[c] / (self.fc * self.clam ** 2 * L[c] ** 2)
            lam = np.interp(-np.log(np.maximum(q, 1e-300)), self._x, self._lam)
            Ac = (self.clam * L[c] / lam) ** 2
            Ac = np.where(q >= 1.0 / self.plateau ** 2, -N[c] / self.fc, Ac)   # stocky: squash load
            A[c] = np.maximum(Ac, (L[c] / (self.lam_c * self.kr)) ** 2)
        t = ~c
        if self.lam_t:
            A[t] = np.maximum(A[t], (L[t] / (self.lam_t * self.kr)) ** 2)
        return np.maximum(A, self.A_min)


# ----------------------------------------------------------------------------------------------- geometry
def seg_cross(P1, P2, Q1, Q2, eps=1e-9):
    """Proper crossing of segments P1P2 and Q1Q2 (broadcast arrays (...,2)); shared endpoints don't count."""
    d1 = P2 - P1
    d2 = Q2 - Q1
    def cr(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    o1 = cr(d1, Q1 - P1)
    o2 = cr(d1, Q2 - P1)
    o3 = cr(d2, P1 - Q1)
    o4 = cr(d2, P2 - Q1)
    s1 = np.maximum(np.abs(d1[..., 0]) + np.abs(d1[..., 1]), 1e-12)
    s2 = np.maximum(np.abs(d2[..., 0]) + np.abs(d2[..., 1]), 1e-12)
    e1 = eps * s1 * s2
    return (o1 * o2 < -e1) & (o3 * o4 < -e1)


def point_seg_dist(P, A, B):
    """Distance from points P (n,2) to segments AB (m,2): returns (n,m) and param t."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    AP = P[:, None, :] - A[None, :, :]
    t = (AP * AB[None]).sum(2) / L2[None]
    tc = np.clip(t, 0, 1)
    proj = A[None] + tc[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(2)), t


def in_poly(P, poly):
    """Even-odd point in polygon (P (n,2), poly (k,2)); boundary counts as inside within 1e-6."""
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    k = len(poly)
    for i in range(k):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % k]
        cond = ((y1 > y) != (y2 > y))
        xint = (x2 - x1) * (y - y1) / ((y2 - y1) if y2 != y1 else 1e-12) + x1
        inside ^= cond & (x < xint)
    # boundary tolerance
    A = np.array(poly, float)
    B = np.roll(A, -1, 0)
    d, _ = point_seg_dist(P, A, B)
    return inside | (d.min(1) < 1e-6)


# ----------------------------------------------------------------------------------------------- brief
class Brief:
    """One engineering brief: fixed nodes (load points and supports), loads, domain, limits, textbook design."""

    def __init__(self, name, title, fixed, supports, cases, domain, steel, L_min, L_max, clear,
                 m_joint, k_free=(6, 18), sym_x=None, unseen=None, textbook=None, notes="",
                 gam_d=1.35, gam_l=1.5, gam_s=1.35, must=None):
        self.name, self.title, self.notes = name, title, notes
        self.fixed = np.array(fixed, float)
        self.nf = len(self.fixed)
        self.supports = supports            # {node: (fix_x, fix_y)}
        self.cases = cases                  # [{'name', 'Fd': (nf,2) N, 'Fl': (nf,2) N, optional 'gd','gl'}]
        self.unseen = unseen or []
        self.rects = np.array(domain, float).reshape(-1, 4)          # union of (x0, y0, x1, y1)
        self.domain = self.rects
        self.steel = steel
        self.L_min, self.L_max, self.clear, self.m_joint = L_min, L_max, clear, m_joint
        self.k_free, self.sym_x = k_free, sym_x
        self.textbook = textbook            # {'nodes': [[x,y],...] (first nf = fixed), 'bars': [[i,j],...]}
        self.gam_d, self.gam_l, self.gam_s = gam_d, gam_l, gam_s
        self.FD = np.array([c['Fd'] for c in cases], float)          # (nc, nf, 2)
        self.FL = np.array([c['Fl'] for c in cases], float)
        self.loaded = np.zeros(self.nf, bool)
        for c in cases + self.unseen:
            self.loaded |= (np.abs(c['Fd']).sum(1) + np.abs(c['Fl']).sum(1)) > 0
        self.must = set(must or []) | set(np.flatnonzero(self.loaded).tolist())
        lo, hi = self.rects[:, :2].min(0), self.rects[:, 2:].max(0)
        self.bbox = (lo, hi)
        self.size = float(np.hypot(*(hi - lo)))
        # genome fixed nodes: the left half when symmetric
        if sym_x is None:
            self.gfixed = np.arange(self.nf)
            self.mirror = None
        else:
            self.gfixed = np.flatnonzero(self.fixed[:, 0] <= sym_x + 1e-6)
            mir = np.full(self.nf, -1)
            for i in range(self.nf):
                p = self.fixed[i].copy()
                p[0] = 2 * sym_x - p[0]
                d = np.hypot(*(self.fixed - p).T)
                j = int(np.argmin(d))
                if d[j] > 1e-6:
                    raise ValueError(f"fixed node {i} has no mirror")
                mir[i] = j
            self.mirror = mir


# ----------------------------------------------------------------------------------------------- structure analysis
class Structure:
    """A decoded design: nodes (first nf are the brief's fixed nodes, unused ones allowed) and bars."""

    def __init__(self, brief, nodes, bars):
        self.b = brief
        nodes = np.asarray(nodes, float)
        bars = np.asarray(bars, int).reshape(-1, 2)
        if len(bars):
            bars = np.unique(np.sort(bars, 1), axis=0)
        # drop nodes without bars (unused supports / fixed nodes), keep index map
        used = np.zeros(len(nodes), bool)
        used[bars.ravel()] = True
        self.ok = True
        self.why = ""
        if np.any(brief.loaded & ~used[:brief.nf]):
            self.ok, self.why = False, "loaded node not connected"
        keep = np.flatnonzero(used)
        remap = -np.ones(len(nodes), int)
        remap[keep] = np.arange(len(keep))
        self.orig = keep                                 # index into the input node list
        self.nodes = nodes[keep]
        self.bars = remap[bars]
        self.n, self.m = len(self.nodes), len(self.bars)
        self.isfixed = keep < brief.nf
        d = self.nodes[self.bars[:, 1]] - self.nodes[self.bars[:, 0]]
        self.L = np.hypot(d[:, 0], d[:, 1])
        self.c, self.s = d[:, 0] / self.L, d[:, 1] / self.L
        nd = 2 * self.n
        self.ndof = nd
        fixed = []
        for i, o in enumerate(keep):
            if o < brief.nf and o in brief.supports:
                fx, fy = brief.supports[o]
                if fx:
                    fixed.append(2 * i)
                if fy:
                    fixed.append(2 * i + 1)
        self.fixed_dofs = np.array(fixed, int)
        self.free = np.setdiff1d(np.arange(nd), self.fixed_dofs)
        self.dofs = np.stack([2 * self.bars[:, 0], 2 * self.bars[:, 0] + 1,
                              2 * self.bars[:, 1], 2 * self.bars[:, 1] + 1], 1)
        self.Bv = np.stack([-self.c, -self.s, self.c, self.s], 1)
        self._rows = np.repeat(self.dofs, 4, axis=1).ravel()
        self._cols = np.tile(self.dofs, (1, 4)).ravel()
        # external loads mapped to the kept nodes
        nc = len(brief.cases)
        self.Fd = np.zeros((nd, nc))
        self.Fl = np.zeros((nd, nc))
        self.gd = np.array([c.get('gd', brief.gam_d) for c in brief.cases])
        self.gl = np.array([c.get('gl', brief.gam_l) for c in brief.cases])
        fx = np.flatnonzero(keep < brief.nf)
        self._fx, self._fo = fx, keep[fx]
        self.Fd[2 * fx, :] = brief.FD[:, self._fo, 0].T
        self.Fd[2 * fx + 1, :] = brief.FD[:, self._fo, 1].T
        self.Fl[2 * fx, :] = brief.FL[:, self._fo, 0].T
        self.Fl[2 * fx + 1, :] = brief.FL[:, self._fo, 1].T
        if self.m == 0 or len(self.fixed_dofs) < 3:
            self.ok, self.why = False, "no supports"

    def _put(self, vec, F):
        vec[2 * self._fx] += F[self._fo, 0]
        vec[2 * self._fx + 1] += F[self._fo, 1]

    def case_loads(self, case):
        F = np.zeros(self.ndof)
        self._put(F, case['Fd'])
        Fl = np.zeros(self.ndof)
        self._put(Fl, case['Fl'])
        return F, Fl

    # -- core solves
    def K(self, A):
        k = self.b.steel.E * A / self.L
        ke = k[:, None, None] * self.Bv[:, :, None] * self.Bv[:, None, :]
        nd = self.ndof
        return np.bincount(self._rows * nd + self._cols, weights=ke.ravel(),
                           minlength=nd * nd).reshape(nd, nd)

    def selfweight(self, A):
        """Nodal self-weight vector (members + joints), N, downward (negative y)."""
        w = self.b.steel.rho * G * A * self.L
        F = np.zeros(self.ndof)
        np.add.at(F, 2 * self.bars[:, 0] + 1, -w / 2)
        np.add.at(F, 2 * self.bars[:, 1] + 1, -w / 2)
        F[1::2] -= self.b.m_joint * 1000.0 * G
        return F

    def stable(self, A):
        Kf = self.K(A)[np.ix_(self.free, self.free)]
        try:
            Lc = np.linalg.cholesky(Kf)
        except np.linalg.LinAlgError:
            return False
        piv = np.diag(Lc) ** 2 / np.maximum(np.diag(Kf), 1e-30)
        return bool(piv.min() > 1e-9)

    def forces(self, A, F):
        """Member forces (tension +) for load matrix F (ndof, nc). Returns None if singular."""
        Kf = self.K(A)[np.ix_(self.free, self.free)]
        try:
            uf = np.linalg.solve(Kf, F[self.free])
        except np.linalg.LinAlgError:
            return None, None
        u = np.zeros((self.ndof,) + F.shape[1:])
        u[self.free] = uf
        k = self.b.steel.E * A / self.L
        if F.ndim == 1:
            N = k * np.einsum("ij,ij->i", self.Bv, u[self.dofs])
        else:
            N = k[:, None] * np.einsum("ij,ijc->ic", self.Bv, u[self.dofs])
        return N, u

    def design_forces(self, A):
        sw = self.selfweight(A)
        F = self.Fd * self.gd + self.Fl * self.gl + np.outer(sw, np.full(len(self.gd), self.b.gam_s))
        return self.forces(A, F)

    def size(self, iters=25, tol=5e-3):
        """Fully stressed design over all design cases. Returns (A, ok)."""
        st = self.b.steel
        if not self.ok:
            return None, False
        A = np.full(self.m, 3 * st.A_min)
        if not self.stable(A):
            return None, False
        for it in range(iters):
            N, _ = self.design_forces(A)
            if N is None or not np.all(np.isfinite(N)):
                return None, False
            Areq = st.area_req(N, self.L[:, None]).max(1)
            ch = np.max(np.abs(Areq - A) / A)
            A = Areq
            if ch < tol:
                break
        # final verification and strict repair: every member utilisation <= 1 in every case
        for _ in range(30):
            u = self.util(A)
            if u is None:
                return None, False
            umax = u.max(1)
            if umax.max() <= 1.0 + 1e-6:
                return A, True
            A = np.where(umax > 1.0, A * np.maximum(umax, 1.0) ** 1.05, A)
        return A, False

    def util(self, A, N=None):
        """Utilisation |N|/capacity per member and design case (m, nc)."""
        if N is None:
            N, _ = self.design_forces(A)
            if N is None:
                return None
        Nt, Nc = self.b.steel.caps(A, self.L)
        return np.where(N >= 0, N / Nt[:, None], -N / Nc[:, None])

    def mass(self, A):
        """tonnes: members + joints."""
        return float(self.b.steel.rho * np.sum(A * self.L) / 1000.0 + self.n * self.b.m_joint)

    def member_mass(self, A):
        return float(self.b.steel.rho * np.sum(A * self.L) / 1000.0)

    # -- ramp to failure
    def ramp(self, A, case, max_events=200):
        """Characteristic dead + self-weight held at 1.0, live load ramped by lambda.
        Members fail one by one (tension: yield; compression: EC3 buckling resistance); a failed member is
        removed (no post-buckling strength assumed). Returns the event list and the peak lambda."""
        alive = np.ones(self.m, bool)
        Fd, Fl = self.case_loads(case)
        st = self.b.steel
        Nt, Nc = st.caps(A, self.L)
        lam_now, peak = 0.0, 0.0
        events = []
        while alive.sum() > 0 and len(events) < max_events:
            sub = Structure.__new__(Structure)
            sub.__dict__.update(self.__dict__)
            idx = np.flatnonzero(alive)
            sub.bars, sub.L, sub.c, sub.s = self.bars[idx], self.L[idx], self.c[idx], self.s[idx]
            sub.m, sub.dofs, sub.Bv = len(idx), self.dofs[idx], self.Bv[idx]
            sub._rows = np.repeat(sub.dofs, 4, axis=1).ravel()
            sub._cols = np.tile(sub.dofs, (1, 4)).ravel()
            Ai = A[idx]
            if not sub.stable(Ai):
                events.append({"bar": -1, "lam": lam_now, "mode": "mechanism"})
                break
            sw = sub.selfweight(Ai)
            Nd, _ = sub.forces(Ai, Fd + sw)
            Nl, _ = sub.forces(Ai, Fl)
            if Nd is None or Nl is None:
                events.append({"bar": -1, "lam": lam_now, "mode": "mechanism"})
                break
            nt, nc = Nt[idx], Nc[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                lt = np.where(Nl > 1e-9, (nt - Nd) / Nl, np.inf)
                lc = np.where(Nl < -1e-9, (nc + Nd) / (-Nl), np.inf)
            # already failing under dead load alone
            dead_fail = (Nd > nt) | (Nd < -nc)
            lt = np.where(dead_fail & (Nd > nt), 0.0, lt)
            lc = np.where(dead_fail & (Nd < -nc), 0.0, lc)
            lam_i = np.minimum(lt, lc)
            k = int(np.argmin(lam_i))
            if not np.isfinite(lam_i[k]):
                break
            lam_e = max(lam_now, float(lam_i[k]))
            mode = "yield" if lt[k] <= lc[k] else "buckle"
            Nfail = float(Nd[k] + lam_e * Nl[k])
            events.append({"bar": int(idx[k]), "lam": lam_e, "mode": mode, "N": Nfail,
                           "cascade": bool(lam_i[k] < lam_now - 1e-9)})
            lam_now = lam_e
            peak = max(peak, lam_e)
            alive[idx[k]] = False
        return events, peak

    def first_fail(self, A, case):
        ev, peak = self.ramp(A, case, max_events=1)
        return ev[0]["lam"] if ev else float("inf")

    def to_json(self, A=None, extra=None):
        d = {"nodes": np.round(self.nodes, 4).tolist(), "bars": self.bars.tolist(),
             "fixed": self.isfixed.tolist(), "orig": self.orig.tolist()}
        if A is not None:
            d["A"] = [float(a) for a in A]
            d["D"] = [float(x) for x in self.b.steel.diameter(A)]
            N, _ = self.design_forces(A)
            if N is not None:
                d["Nmax"] = [float(x) for x in N.max(1)]
                d["Nmin"] = [float(x) for x in N.min(1)]
                d["util"] = [float(x) for x in self.util(A, N).max(1)]
            d["mass_t"] = self.mass(A)
        if extra:
            d.update(extra)
        return d


# ----------------------------------------------------------------------------------------------- genome
class Genome:
    """Free nodes (positions) + bars over [genome fixed nodes, free nodes] (left half when symmetric).
    Node ids in bars: 0..nf-1 = the brief's fixed nodes (only genome-fixed ones are used), nf.. = free nodes."""
    __slots__ = ("pos", "bars", "clean")

    def __init__(self, pos, bars):
        self.pos = np.asarray(pos, float).reshape(-1, 2)
        self.bars = [tuple(sorted(b)) for b in bars]
        self.clean = True

    def copy(self):
        return Genome(self.pos.copy(), list(self.bars))

    def key(self):
        return (np.round(self.pos, 3).tobytes(), tuple(sorted(self.bars)))


def genome_nodes(brief, g):
    return np.vstack([brief.fixed, g.pos]) if len(g.pos) else brief.fixed.copy()


def decode(brief, g):
    """Genome -> full Structure (mirrored when symmetric)."""
    nf = brief.nf
    if brief.sym_x is None:
        return Structure(brief, genome_nodes(brief, g), g.bars)
    xm = brief.sym_x
    pos = g.pos
    on_axis = np.abs(pos[:, 0] - xm) < 1e-6 if len(pos) else np.zeros(0, bool)
    # full node list: fixed (all), free (left/axis), mirrored free (non-axis)
    mir_free = pos[~on_axis].copy()
    mir_free[:, 0] = 2 * xm - mir_free[:, 0]
    nodes = np.vstack([brief.fixed, pos, mir_free]) if len(pos) else brief.fixed.copy()
    k = len(pos)
    mid = -np.ones(k, int)
    mid[~on_axis] = nf + k + np.arange((~on_axis).sum())
    mid[on_axis] = nf + np.flatnonzero(on_axis)

    def mirror(i):
        if i < nf:
            return int(brief.mirror[i])
        return int(mid[i - nf])
    bars = set(g.bars)
    for a, b in g.bars:
        ma, mb = mirror(a), mirror(b)
        bars.add(tuple(sorted((ma, mb))))
    return Structure(brief, nodes, sorted(bars))


def violations(brief, S):
    """Independent check of a decoded structure: crossings, joint clearance, member lengths, domain."""
    P1, P2 = S.nodes[S.bars[:, 0]], S.nodes[S.bars[:, 1]]
    X = seg_cross(P1[:, None, :], P2[:, None, :], P1[None, :, :], P2[None, :, :])
    np.fill_diagonal(X, False)
    dist, t = point_seg_dist(S.nodes, P1, P2)
    near = (dist < brief.clear * 0.999) & (t > 1e-6) & (t < 1 - 1e-6)
    ids = np.arange(S.n)[:, None]
    near &= (ids != S.bars[None, :, 0]) & (ids != S.bars[None, :, 1])
    ts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    Smp = (P1[:, None, :] * (1 - ts)[None, :, None] + P2[:, None, :] * ts[None, :, None]).reshape(-1, 2)
    out = (~in_domain(brief, Smp)).sum()
    return {"crossings": int(X.sum() // 2), "clearance": int(near.sum()),
            "short": int((S.L < brief.L_min - 1e-6).sum()), "long": int((S.L > brief.L_max + 1e-6).sum()),
            "outside": int(out)}


def in_domain(brief, P, eps=1e-6):
    """Domain = union of axis-aligned rectangles (x0, y0, x1, y1)."""
    P = np.asarray(P, float).reshape(-1, 2)
    R = brief.rects
    x, y = P[:, 0:1], P[:, 1:2]
    ok = (x >= R[None, :, 0] - eps) & (x <= R[None, :, 2] + eps) & (y >= R[None, :, 1] - eps) & (y <= R[None, :, 3] + eps)
    return ok.any(1)


def repair(brief, g, max_pass=25):
    """Enforce the geometric rules on a genome (in place) and return it:
    nodes inside the domain, merge nodes closer than L_min, bars within [L_min, L_max] and inside the domain,
    a bar passing through a joint is split there, crossing bars: the longer one is removed,
    free nodes need >= 2 bars, a free node with two nearly collinear bars is dissolved."""
    nf = brief.nf
    gmask = np.zeros(nf, bool)
    gmask[brief.gfixed] = True
    xm = brief.sym_x
    lo, hi = brief.bbox
    ts = np.linspace(0.1, 0.9, 9)
    clean = False
    for _ in range(max_pass):
        changed = False
        # 0. clamp, snap to the axis
        if len(g.pos):
            g.pos[:, 0] = np.clip(g.pos[:, 0], lo[0], hi[0] if xm is None else xm)
            g.pos[:, 1] = np.clip(g.pos[:, 1], lo[1], hi[1])
            if xm is not None:
                snap = np.abs(g.pos[:, 0] - xm) < 0.25 * brief.L_min
                g.pos[snap, 0] = xm
            # 1. free nodes outside the domain
            ins = in_domain(brief, g.pos)
            if not ins.all():
                _drop_free(g, np.flatnonzero(~ins), nf)
                changed = True
        # 2. merge free nodes closer than L_min to an earlier node (fixed nodes first)
        k = len(g.pos)
        if k:
            P = genome_nodes(brief, g)
            valid = np.concatenate([gmask, np.ones(k, bool)])
            D = np.hypot(P[:, None, 0] - P[None, nf:, 0], P[:, None, 1] - P[None, nf:, 1])   # (all, free)
            D[~valid] = np.inf
            target = np.arange(nf + k)
            merged = []
            for j in range(k):
                jj = nf + j
                col = D[:jj, j]
                cand = np.flatnonzero(col < brief.L_min)
                cand = [c for c in cand if target[c] == c]
                if cand:
                    i = int(min(cand, key=lambda c: col[c]))
                    target[jj] = i
                    merged.append(j)
            if merged:
                bars = []
                for a_, b_ in g.bars:
                    a2, b2 = int(target[a_]), int(target[b_])
                    if a2 != b2:
                        bars.append(tuple(sorted((a2, b2))))
                g.bars = bars
                _drop_free(g, merged, nf)
                changed = True
        # 3. bars: dedupe, allowed fixed nodes, length limits, inside the domain
        nodes = genome_nodes(brief, g)
        if g.bars:
            B = np.array(sorted(set(g.bars)), int)
            B = B[B[:, 0] != B[:, 1]]
            okf = np.ones(len(B), bool)
            okf &= np.where(B[:, 0] < nf, gmask[np.minimum(B[:, 0], nf - 1)], True)
            okf &= np.where(B[:, 1] < nf, gmask[np.minimum(B[:, 1], nf - 1)], True)
            P1, P2 = nodes[B[:, 0]], nodes[B[:, 1]]
            L = np.hypot(*(P2 - P1).T)
            okf &= (L >= brief.L_min - 1e-9) & (L <= brief.L_max + 1e-9)
            S = (P1[:, None, :] * (1 - ts)[None, :, None] + P2[:, None, :] * ts[None, :, None]).reshape(-1, 2)
            okf &= in_domain(brief, S).reshape(len(B), -1).all(1)
            if len(B) != len(g.bars) or not okf.all():
                changed = True
            B = B[okf]
            g.bars = [tuple(x) for x in B.tolist()]
        else:
            B = np.zeros((0, 2), int)
        if len(B):
            P1, P2 = nodes[B[:, 0]], nodes[B[:, 1]]
            # 4. a bar passing a joint -> split it at that joint (one per bar per pass)
            used = np.unique(B.ravel())
            dist, t = point_seg_dist(nodes[used], P1, P2)
            mask = (dist < brief.clear) & (t > 1e-6) & (t < 1 - 1e-6)
            mask &= (used[:, None] != B[None, :, 0]) & (used[:, None] != B[None, :, 1])
            if mask.any():
                bars = set(g.bars)
                for bi in np.flatnonzero(mask.any(0)):
                    ni = np.flatnonzero(mask[:, bi])
                    n = int(used[ni[np.argmin(t[ni, bi])]])
                    a_, b_ = int(B[bi, 0]), int(B[bi, 1])
                    bars.discard((a_, b_))
                    bars.add(tuple(sorted((a_, n))))
                    bars.add(tuple(sorted((n, b_))))
                g.bars = sorted(bars)
                changed = True
                continue
            # 5. crossings: remove the longer bar of each crossing pair, longest first
            if len(B) > 1:
                X = seg_cross(P1[:, None, :], P2[:, None, :], P1[None, :, :], P2[None, :, :])
                np.fill_diagonal(X, False)
                if X.any():
                    Lb = np.hypot(*(P2 - P1).T)
                    alive = np.ones(len(B), bool)
                    for i in np.argsort(-Lb):
                        if alive[i] and (X[i] & alive).any():
                            alive[i] = False
                    g.bars = [tuple(x) for x in B[alive].tolist()]
                    changed = True
                    continue
        # 6. free nodes need >= 2 bars (axis nodes >= 1, their mirror adds one); collinear degree-2 dissolve
        k = len(g.pos)
        if k:
            deg = np.zeros(nf + k, int)
            if g.bars:
                Bb = np.array(g.bars)
                np.add.at(deg, Bb.ravel(), 1)
            fdeg = deg[nf:]
            axis = np.abs(g.pos[:, 0] - xm) < 1e-6 if xm is not None else np.zeros(k, bool)
            low = np.flatnonzero((fdeg < 2) & ~(axis & (fdeg >= 1)))
            if len(low):
                _drop_free(g, low, nf)
                changed = True
                continue
            two = np.flatnonzero((fdeg == 2) & ~axis)
            if len(two):
                nodes = genome_nodes(brief, g)
                Bb = np.array(g.bars)
                for i in two:
                    ii = nf + i
                    rows = np.flatnonzero((Bb[:, 0] == ii) | (Bb[:, 1] == ii))
                    nb = [int(Bb[r, 1] if Bb[r, 0] == ii else Bb[r, 0]) for r in rows]
                    v1, v2 = nodes[nb[0]] - nodes[ii], nodes[nb[1]] - nodes[ii]
                    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                    if cosang < -0.996:
                        g.bars = [bb for bb in g.bars if ii not in bb] + [tuple(sorted(nb))]
                        _drop_free(g, [i], nf)
                        changed = True
                        break
                if changed:
                    continue
        if not changed:
            clean = True
            break
    g.clean = clean
    return g


def _drop_free(g, idx, nf):
    idx = sorted(set(int(i) for i in idx))
    keep = np.setdiff1d(np.arange(len(g.pos)), idx)
    remap = {}
    for new, old in enumerate(keep):
        remap[nf + old] = nf + new
    bars = []
    for a, b in g.bars:
        if (a >= nf and a not in remap) or (b >= nf and b not in remap):
            continue
        a2 = remap.get(a, a) if a >= nf else a
        b2 = remap.get(b, b) if b >= nf else b
        bars.append(tuple(sorted((a2, b2))))
    g.pos = g.pos[keep]
    g.bars = bars


def _merge(g, src, dst, nf):
    """Merge node src (free) into dst: bars re-pointed, then src dropped."""
    bars = []
    for a, b in g.bars:
        a2 = dst if a == src else a
        b2 = dst if b == src else b
        if a2 != b2:
            bars.append(tuple(sorted((a2, b2))))
    g.bars = list(dict.fromkeys(bars))
    _drop_free(g, [src - nf], nf)


def greedy_triangulation(brief, P_free, rng=None):
    """Bars for a random node cloud: all pairs within limits, accepted shortest first unless crossing."""
    nf = brief.nf
    ids = np.array([int(i) for i in brief.gfixed] + list(range(nf, nf + len(P_free))))
    nodes = np.vstack([brief.fixed, P_free]) if len(P_free) else brief.fixed
    P = nodes[ids]
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    kn = min(10, len(ids) - 1)
    nn = np.argsort(D, 1)[:, :kn]
    pairs = {(min(i, int(j)), max(i, int(j))) for i in range(len(ids)) for j in nn[i]}
    pr = np.array(sorted(pairs), int).reshape(-1, 2)
    ii, jj = pr[:, 0], pr[:, 1]
    L = D[ii, jj]
    ok = (L >= brief.L_min) & (L <= min(brief.L_max, 0.35 * brief.size + brief.L_min))
    ii, jj, L = ii[ok], jj[ok], L[ok]
    order = np.argsort(L)
    ii, jj = ii[order], jj[order]
    A1, A2 = P[ii], P[jj]
    n = len(ii)
    acc1 = np.zeros((n, 2))
    acc2 = np.zeros((n, 2))
    na = 0
    acc = []
    for k in range(n):
        if na and seg_cross(A1[k][None], A2[k][None], acc1[:na], acc2[:na]).any():
            continue
        acc.append((int(ids[ii[k]]), int(ids[jj[k]])))
        acc1[na] = A1[k]
        acc2[na] = A2[k]
        na += 1
    return acc


# ----------------------------------------------------------------------------------------------- evaluation
INFEAS = 1e6


def evaluate(brief, g):
    """Returns (fitness tonnes, A or None, Structure)."""
    S = decode(brief, g)
    if not S.ok or not getattr(g, "clean", True):
        return INFEAS + 1e4, None, S
    A, ok = S.size()
    if not ok:
        return INFEAS + 1e3, None, S
    return S.mass(A), A, S


def cleanup(brief, S, A, rel=0.04):
    """Disclosed cleanup rule on a final design: repeatedly remove the member with the lowest peak
    utilisation (if below `rel`) when the structure stays stable, all checks pass and total mass drops."""
    nodes, bars = S.nodes.copy(), S.bars.copy()
    # rebuild in brief node space: fixed nodes first
    full = np.vstack([brief.fixed, nodes[~S.isfixed]])
    remap = np.empty(S.n, int)
    remap[S.isfixed] = S.orig[S.isfixed]
    remap[~S.isfixed] = brief.nf + np.arange((~S.isfixed).sum())
    bars = remap[bars]
    best = Structure(brief, full, bars)
    bestA, ok = best.size()
    best_m = best.mass(bestA) if ok else INFEAS
    removed = 0
    while True:
        u = best.util(bestA).max(1)
        order = np.argsort(u)
        improved = False
        for i in order[:6]:
            if u[i] > rel:
                break
            nb = np.delete(best.bars, i, 0)
            fullb = np.vstack([brief.fixed, best.nodes[~best.isfixed]])
            rm = np.empty(best.n, int)
            rm[best.isfixed] = best.orig[best.isfixed]
            rm[~best.isfixed] = brief.nf + np.arange((~best.isfixed).sum())
            T = Structure(brief, fullb, rm[nb])
            TA, ok = T.size()
            if ok and T.mass(TA) < best_m - 1e-6:
                best, bestA, best_m = T, TA, T.mass(TA)
                removed += 1
                improved = True
                break
        if not improved:
            break
    return best, bestA, removed


# ----------------------------------------------------------------------------------------------- evolution
def random_genome(brief, rng):
    lo, hi = brief.bbox
    hix = hi[0] if brief.sym_x is None else brief.sym_x
    k = int(rng.integers(brief.k_free[0], brief.k_free[1] + 1))
    pts = []
    tries = 0
    while len(pts) < k and tries < 50 * k:
        tries += 1
        p = np.array([rng.uniform(lo[0], hix), rng.uniform(lo[1], hi[1])])
        if in_domain(brief, p[None])[0]:
            pts.append(p)
    P = np.array(pts).reshape(-1, 2)
    bars = greedy_triangulation(brief, P, rng)
    g = Genome(P, bars)
    return repair(brief, g)


def mutate(brief, parent, rng, util_hint=None):
    g = parent.copy()
    nf = brief.nf
    lo, hi = brief.bbox
    hix = hi[0] if brief.sym_x is None else brief.sym_x
    nops = 1 + rng.poisson(0.6)
    ops = []
    for _ in range(nops):
        r = rng.random()
        k = len(g.pos)
        nodes = genome_nodes(brief, g)
        if r < 0.34 and k:                                   # move 1-3 nodes, heavy-tailed step
            ops.append("move")
            sel = rng.choice(k, size=min(k, 1 + rng.poisson(0.5)), replace=False)
            sig = brief.size * 10 ** rng.uniform(-2.6, -0.8)
            g.pos[sel] += rng.normal(0, sig, (len(sel), 2))
        elif r < 0.40 and k:                                 # nudge every free node a little
            ops.append("jiggle")
            g.pos += rng.normal(0, brief.size * 0.003, g.pos.shape)
        elif r < 0.52:                                       # add a bar from a node to a nearby node
            ops.append("add_bar")
            ids = list(int(i) for i in brief.gfixed) + list(range(nf, nf + k))
            a = int(rng.choice(ids))
            d = np.hypot(*(nodes[ids] - nodes[a]).T)
            cand = [ids[j] for j in np.argsort(d)[1:10] if brief.L_min <= d[j] <= brief.L_max]
            cand = [c for c in cand if tuple(sorted((a, c))) not in set(g.bars)]
            if cand:
                c = int(cand[min(len(cand) - 1, int(rng.exponential(1.5)))])
                g.bars.append(tuple(sorted((a, c))))
        elif r < 0.64 and g.bars:                            # remove a bar (half the time the least used one)
            ops.append("remove_bar")
            if util_hint is not None and len(util_hint) == len(g.bars) and rng.random() < 0.5:
                i = int(np.argmin(util_hint + rng.random(len(util_hint)) * 0.02))
            else:
                i = int(rng.integers(len(g.bars)))
            g.bars.pop(i)
        elif r < 0.75 and g.bars:                            # split a bar with a new joint, braced to a neighbour
            ops.append("split")
            i = int(rng.integers(len(g.bars)))
            a, b = g.bars[i]
            t = rng.uniform(0.3, 0.7)
            p = nodes[a] * (1 - t) + nodes[b] * t + rng.normal(0, brief.size * 0.01, 2)
            p[0] = min(max(p[0], lo[0]), hix)
            p[1] = min(max(p[1], lo[1]), hi[1])
            n = nf + k
            g.pos = np.vstack([g.pos, p])
            g.bars.pop(i)
            g.bars += [tuple(sorted((a, n))), tuple(sorted((n, b)))]
            ids = list(int(x) for x in brief.gfixed) + list(range(nf, nf + k))
            d = np.hypot(*(nodes[ids] - p).T)
            for j in np.argsort(d):
                c = ids[j]
                if c not in (a, b) and brief.L_min <= d[j] <= brief.L_max:
                    g.bars.append(tuple(sorted((n, c))))
                    break
        elif r < 0.82:                                       # new joint tied to its 3 nearest nodes
            ops.append("add_node")
            p = np.array([rng.uniform(lo[0], hix), rng.uniform(lo[1], hi[1])])
            n = nf + k
            ids = list(int(x) for x in brief.gfixed) + list(range(nf, nf + k))
            d = np.hypot(*(nodes[ids] - p).T)
            g.pos = np.vstack([g.pos, p])
            near = [ids[j] for j in np.argsort(d) if brief.L_min <= d[j] <= brief.L_max][:3]
            g.bars += [tuple(sorted((n, c))) for c in near]
        elif r < 0.90 and k:                                 # remove a joint: merge it into its nearest node
            ops.append("remove_node")
            i = int(rng.integers(k))
            ii = nf + i
            ids = list(int(x) for x in brief.gfixed) + [nf + j for j in range(k) if j != i]
            d = np.hypot(*(nodes[ids] - nodes[ii]).T)
            _merge(g, ii, int(ids[int(np.argmin(d))]), nf)
        elif g.bars:                                         # flip: swap a shared triangle edge
            ops.append("flip")
            adj = {}
            for a, b in g.bars:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            for _t in range(6):
                a, b = g.bars[int(rng.integers(len(g.bars)))]
                common = list(adj.get(a, set()) & adj.get(b, set()))
                if len(common) >= 2:
                    c, d_ = common[:2]
                    g.bars.remove((a, b))
                    g.bars.append(tuple(sorted((c, d_))))
                    break
    repair(brief, g)
    return g, ops


def tournament(fit, k, rng):
    idx = rng.integers(0, len(fit), k)
    return int(idx[np.argmin(fit[idx])])


def evolve(brief, seed=1, pop=160, gens=400, elite=4, tour=3, sample_every=0, sample_n=0,
           log=None, time_limit=None):
    """Mutation + selection, no crossover, so every design has exactly one parent (a true family tree).
    Returns a dict with the winner genome, history and the lineage arrays."""
    import time as _t
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(seed + 10_000)        # sampling for the ghost visuals never touches rng
    t0 = _t.perf_counter()
    lin_id, lin_par, lin_gen, lin_fit = [], [], [], []
    genomes = {}
    nid = 0
    P, F, U, ID = [], [], [], []
    for i in range(pop):
        g = random_genome(brief, rng)
        f, A, S = evaluate(brief, g)
        P.append(g)
        F.append(f)
        U.append(_util_hint(brief, g, S, A))
        ID.append(nid)
        genomes[nid] = g
        lin_id.append(nid); lin_par.append(-1); lin_gen.append(0); lin_fit.append(f)
        nid += 1
    F = np.array(F)
    hist = []
    samples = []
    popids = [[int(x) for x in ID]]
    for gen in range(1, gens + 1):
        order = np.argsort(F)
        newP, newF, newU, newID = [], [], [], []
        for i in order[:elite]:
            newP.append(P[i]); newF.append(F[i]); newU.append(U[i]); newID.append(ID[i])
        while len(newP) < pop:
            pi = tournament(F, tour, rng)
            child, ops = mutate(brief, P[pi], rng, U[pi])
            f, A, S = evaluate(brief, child)
            newP.append(child); newF.append(f); newU.append(_util_hint(brief, child, S, A)); newID.append(nid)
            genomes[nid] = child
            lin_id.append(nid); lin_par.append(ID[pi]); lin_gen.append(gen); lin_fit.append(f)
            nid += 1
        P, F, U, ID = newP, np.array(newF), newU, newID
        fe = F[F < INFEAS]
        hist.append({"gen": gen, "best": float(F.min()), "median": float(np.median(fe)) if len(fe) else None,
                     "feasible": float((F < INFEAS).mean()), "best_id": int(ID[int(np.argmin(F))])})
        popids.append([int(x) for x in ID])
        if sample_every and gen % sample_every == 0:
            pick = srng.choice(pop, size=min(sample_n, pop), replace=False)
            samples.append({"gen": gen, "ids": [int(ID[i]) for i in pick]})
            for i in pick:
                genomes[ID[i]] = P[i]
        if gen % 20 == 0:
            keep = _ancestors(lin_par, ID)
            if samples:
                for s in samples:
                    keep.update(s["ids"])
            for k_ in list(genomes):
                if k_ not in keep:
                    del genomes[k_]
        if log and gen % 25 == 0:
            log(f"{brief.name} s{seed} gen {gen}: best {F.min():.2f} t, median {np.median(fe) if len(fe) else -1:.2f}, "
                f"feasible {100 * (F < INFEAS).mean():.0f}%, {_t.perf_counter() - t0:.0f}s")
        if time_limit and _t.perf_counter() - t0 > time_limit:
            break
    bi = int(np.argmin(F))
    return {"winner": P[bi], "winner_id": int(ID[bi]), "fitness": float(F[bi]), "hist": hist,
            "lineage": (np.array(lin_id), np.array(lin_par), np.array(lin_gen), np.array(lin_fit)),
            "genomes": genomes, "samples": samples, "popids": popids, "seconds": _t.perf_counter() - t0}


def _util_hint(brief, g, S, A):
    """Peak utilisation per genome bar (for the 'remove the least used bar' mutation)."""
    if A is None:
        return None
    u = S.util(A)
    if u is None:
        return None
    u = u.max(1)
    # map structure bars back to genome bars (structure bars include mirrors; take the max)
    nf = brief.nf
    idx = {}
    for k, (a, b) in enumerate(S.bars):
        oa, ob = S.orig[a], S.orig[b]
        idx[tuple(sorted((int(oa), int(ob))))] = u[k]
    out = []
    for a, b in g.bars:
        out.append(idx.get(tuple(sorted((a, b))), 1.0))
    return np.array(out)


def _ancestors(par, ids):
    par = np.asarray(par)
    keep = set()
    stack = list(ids)
    while stack:
        i = stack.pop()
        if i in keep or i < 0:
            continue
        keep.add(i)
        stack.append(int(par[i]))
    return keep


def ancestry(par, wid):
    chain = []
    i = wid
    while i >= 0:
        chain.append(int(i))
        i = int(par[i])
    return chain[::-1]


def textbook_structure(brief):
    tb = brief.textbook
    return Structure(brief, np.array(tb["nodes"], float), np.array(tb["bars"], int))


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

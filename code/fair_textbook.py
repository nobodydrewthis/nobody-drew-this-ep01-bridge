"""Fairness check: give the textbook design its own best proportions under the same rules, so evolution is
compared with the strongest version of the human design, not a strawman.
Usage: py -I fair_textbook.py <brief>"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ndt_engine as E
import briefs as BR
from run_candidates import calibrate_joint


def pratt(brief, h, span=40.0, p=4.0):
    xs = np.arange(0, span + 1e-9, p)
    nf = len(xs)
    nodes = [tuple(x) for x in brief.fixed] + [(x, h) for x in xs[1:-1]]
    T = nf
    ntop = nf - 2
    bars = [(i, i + 1) for i in range(nf - 1)] + [(T + i, T + i + 1) for i in range(ntop - 1)]
    bars += [(0, T), (nf - 1, T + ntop - 1)] + [(i, T + i - 1) for i in range(1, nf - 1)]
    mid = nf // 2
    bars += [(T + i - 1, i + 1) for i in range(1, mid)] + [(T + i - 1, i - 1) for i in range(mid + 1, nf - 1)]
    return nodes, bars


def warren_v(brief, h, span=30.0, p=3.0):
    xs = np.arange(0, span + 1e-9, p)
    nf = len(xs)
    nodes = [tuple(x) for x in brief.fixed] + [(x, h) for x in xs[1:-1]]
    T = nf
    ntop = nf - 2
    bars = [(i, i + 1) for i in range(nf - 1)] + [(T + i, T + i + 1) for i in range(ntop - 1)]
    bars += [(0, T), (nf - 1, T + ntop - 1)] + [(i, T + i - 1) for i in range(1, nf - 1)]
    for i in range(1, nf - 2):
        bars.append((i, T + i) if i % 2 == 1 else (i + 1, T + i - 1))
    return nodes, bars


BUILDERS = {"bridge40": (pratt, np.arange(3.0, 8.01, 0.25)), "rail30": (warren_v, np.arange(3.0, 7.01, 0.25))}

if __name__ == "__main__":
    name = sys.argv[1]
    b = BR.ALL[name]()
    mj = calibrate_joint(b)
    build, hs = BUILDERS[name]
    best = None
    for h in hs:
        nodes, bars = build(b, h)
        S = E.Structure(b, np.array(nodes), np.array(bars))
        A, ok = S.size()
        m = S.mass(A) if ok else float("nan")
        print(f"depth {h:5.2f} m: {m:.3f} t" + ("  (standard)" if abs(h - 5.0) < 1e-9 or abs(h - 4.5) < 1e-9 else ""))
        if ok and (best is None or m < best[1]):
            best = (h, m)
    print(f"BEST textbook depth {best[0]:.2f} m -> {best[1]:.3f} t (joint {mj * 1000:.1f} kg)")

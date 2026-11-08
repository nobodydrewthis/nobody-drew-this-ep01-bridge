"""Ghost family tree, generational view: ring g holds the 160 designs alive in generation g. Each design sits next to
its parent from ring g-1 (survivors are their own parent), so families stay together and you can watch one family
take over the whole ring. Writes <out>.bin: per (gen, slot) float32 [angle, parent_angle, fitness_rank, flags, id]
and <out>.json (meta + the winner's line through the rings).
Usage: py -I tree_layout2.py <lineage.npz> <winner_id> <out_prefix>"""
import json
import sys
import numpy as np

lin = np.load(sys.argv[1])
wid = int(sys.argv[2])
out = sys.argv[3]
par, gen, fit, pops = lin["parent"], lin["gen"], lin["fit"], lin["popids"]
G1, P = pops.shape                      # generations 0..G, population size
feas = fit < 1e5
fmin, fmax = fit[feas].min(), np.percentile(fit[feas], 99)

pos = {}                                # id -> slot rank in its current ring (updated each generation)
rows = np.zeros((G1, P, 5), np.float32)
order0 = sorted(range(P), key=lambda s: fit[pops[0][s]])
ring_prev = [int(pops[0][s]) for s in order0]
for r, i in enumerate(ring_prev):
    pos[i] = r


def ang(rank):
    return 2 * np.pi * (rank + 0.5) / P


for r, i in enumerate(ring_prev):
    rows[0, r] = (ang(r), ang(r), 0, 0, i)
K = np.arange(P)
OFF = (K[None, :] + K[:, None]) % P                  # OFF[o, k] = slot of the k-th child for cyclic offset o
for g in range(1, G1):
    cur = [int(x) for x in pops[g]]
    prev_slot = {i: s for s, i in enumerate(ring_prev) if i is not None}

    def gp(i):
        return i if i in prev_slot else int(par[i])
    # children in their parents' order; within one parent, a fixed pseudo-random order
    cur.sort(key=lambda i: (prev_slot[gp(i)], (i * 2654435761) % 1000003))
    ps = np.array([prev_slot[gp(i)] for i in cur])
    # The ring is a circle, so place the ordered children with the cyclic offset that keeps them closest to their
    # parents. A fixed start slot pins the drift to zero there, no line ever crosses it, and the additive drawing
    # showed that spot as a dark radial seam through every generation.
    d = (OFF - ps[None, :]) % P
    cost = np.minimum(d, P - d).sum(1) + 1e-3 * np.minimum(K, P - K)
    o = int(np.argmin(cost))
    ring = [None] * P
    for k, i in enumerate(cur):
        s = int((k + o) % P)
        ring[s] = i
        rank_fit = 0.0 if not feas[i] else float((fmax - min(fit[i], fmax)) / max(fmax - fmin, 1e-9))
        flags = (2.0 if not feas[i] else 0.0) + (4.0 if gp(i) == i else 0.0)
        rows[g, s] = (ang(s), ang(int(ps[k])), rank_fit, flags, i)
    ring_prev = ring
# winner's line: from the last ring back through generational parents
line = []
i = wid
for g in range(G1 - 1, -1, -1):
    ring = rows[g, :, 4].astype(np.int64)
    k = int(np.flatnonzero(ring == i)[0])
    line.append((g, k))
    if g > 0:
        prevring = set(rows[g - 1, :, 4].astype(np.int64).tolist())
        i = i if i in prevring else int(par[i])
line = line[::-1]
for g, k in line:
    rows[g, k, 3] += 1.0
rot = -np.pi / 2 - rows[G1 - 1, line[-1][1], 0]           # winner at the top
rows[:, :, 0] += rot
rows[:, :, 1] += rot
rows.astype(np.float32).tofile(out + ".bin")
# how much of each ring belongs to the winner's founder family
founder = line[0]
fam = []
anc_cache = {}


def founder_of(i):
    j = i
    path = []
    while par[j] >= 0 and j not in anc_cache:
        path.append(j)
        j = int(par[j])
    f = anc_cache.get(j, j)
    for p in path:
        anc_cache[p] = f
    return f


wf = founder_of(wid)
for g in range(G1):
    ids = rows[g, :, 4].astype(np.int64)
    fam.append(float(np.mean([founder_of(int(i)) == wf for i in ids])))
meta = {"generations": int(G1 - 1), "pop": int(P), "winner": wid, "line": [[int(g), int(k)] for g, k in line],
        "line_fit": [float(fit[int(rows[g, k, 4])]) for g, k in line], "winner_family_share": fam,
        "fields": ["angle", "parent_angle", "fit_rank", "flags(1 line, 2 infeasible, 4 survivor)", "id"]}
json.dump(meta, open(out + ".json", "w"))
print(f"rings {G1}, pop {P}; winner's family holds {100 * fam[50]:.0f}% of ring 50, {100 * fam[-1]:.0f}% of the last ring")

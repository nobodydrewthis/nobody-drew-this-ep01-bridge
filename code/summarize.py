"""Table of every candidate run (mass vs textbook, equal-mass load factors, unseen loads, look), as markdown.
Usage: py -I summarize.py <candidates_dir> > table.md"""
import glob
import json
import os
import sys

src = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(src, "*_s[0-9].json"))):
    r = json.load(open(p))
    a = r["analysis"]
    un = a["unseen"][0] if a["unseen"] else None
    rows.append((r["brief"], r["seed"], a, un, r))
print("| brief | seed | evolved t | textbook t | mass | equal-mass worst case | beats every case | unseen first fail (evolved / textbook) | members | joints | shared | gens | min |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for b, s, a, un, r in rows:
    uns = f"x{un['first_evolved']:.2f} / x{un['first_textbook']:.2f}" if un else "-"
    print(f"| {b} | {s} | {a['mass_t']:.4g} | {a['textbook_mass_t']:.4g} | {100 * (a['mass_ratio'] - 1):+.1f}% | "
          f"x{a['worst_case_ratio']:.2f} | {'yes' if a['beats_all_cases'] else 'NO'} | {uns} | {a['bars']} | {a['joints']} | "
          f"{100 * a['shared_bars']:.0f}% | {len(r['hist'])} | {r['seconds'] / 60:.1f} |")
print()
for b, s, a, un, r in rows:
    v = a["violations"]
    bad = {k: x for k, x in v.items() if x}
    if bad:
        print(f"VIOLATIONS {b} s{s}: {bad}")

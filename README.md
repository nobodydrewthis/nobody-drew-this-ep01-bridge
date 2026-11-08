# Nobody Drew This · episode 1: the bridge

The code, the seeds and every result behind the episode *Nobody Drew This · episode 1*.

## What this is

A 40 m, two-lane steel road bridge is designed twice under the same rules (`PROTOCOL.md`):

- once by hand, as a textbook Pratt truss at its best depth;
- once by **evolutionary search**: a population of 160 random truss layouts. Each generation, every design is sized and weighed and checked against the rules. The lighter designs win small tournaments, and each child is a copy of a winner with one random change. Nobody draws the result.

The solver is a 2D pin-jointed truss model. Members are steel tubes, checked for yield in tension and for buckling in compression (Eurocode 3, buckling curve a) under three traffic cases: the whole bridge full, the left half and the right half. Each design is fully stressed, so every tube is as thin as the rules allow.

**Disclosure, as shown in the video:** the numbers come from solver physics. The 3D collapse is visualised from the solver's first failure. It is a picture of the result, not a separate physical test.

This is a demonstration of an algorithm under fixed rules, not a bridge design. Its limits are said in the video:

- 2D;
- linear analysis to the first member failure;
- no fatigue;
- idealised joints;
- continuous member sizes.

## Run it

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
cd code
python episode_analysis.py        # bridge40 seed 1 vs the Pratt at its best depth -> ../data/episode.json
python pynite_check.py            # the same designs in an independent solver (PyNiteFEA) -> ../data/pynite_check.json
python run_candidates.py ../data/rerun 10          # all 7 briefs x seeds 1 and 2 (about 10 min on 10 cores)
python rerun_full.py ../data/candidates/bridge40_s1.json ../data/rerun_bridge40_s1   # the episode run, with its full lineage
```

`episode_analysis.py` prints `evolved 6.710 t | Pratt best depth 6.00 m 8.961 t`, the episode's 6.71 t against 8.96 t.

The seeds are 1 and 2 for every brief (`numpy.random.default_rng(seed)`, `seeds.json`). The same seed and the same code give the same winner, bit for bit. `data/run_bridge40_s1/run.json` records that check as `"reproduced": true`.

## Which file backs each number in the video

| In the video | Where |
|---|---|
| 8.96 t of steel (the Pratt at its best depth, 6 m) | `data/episode.json`: `mass_pratt_best_t`, `pratt_best_depth_m` |
| 6.71 t (the evolved bridge) | `data/episode.json`: `mass_evolved_t` |
| 25 % less steel | `data/episode.json`: `saving_vs_best_pratt` |
| 160 random bridges, 12 pass every check | `data/run_bridge40_s1/generation0.json` |
| 400 generations, 62,560 designs | `data/run_bridge40_s1/run.json`: `generations`, `evals` |
| about 6½ minutes on one laptop core | `data/run_bridge40_s1/run.json`: `seconds` (396 s) |
| a random founder of 10.45 t, then 100 ancestors | `data/run_bridge40_s1/ancestry.json` |
| 214.1 t (Pratt) and 348.2 t (evolved): first tube buckles, left half loaded, equal steel | `data/episode.json`: `cases` ("traffic left half", `first_lam`) × `live_t_per_lambda` / 2. The failing members are `events[0].bar` (Pratt 14, evolved 20). |
| 1.63× what the Pratt held | `data/episode.json`: `cases[...].ratio_first` |
| a 180 t transformer: Pratt 90 t, evolved 161 t | `data/episode.json`: `unseen` (`first_lam` × 180 t) |
| a second, independent solver: 0.000 % apart | `data/pynite_check.json`: `pinned` (`max_rel_diff`) |
| welded joints: ×1.75 full traffic, ×1.45 half traffic | `data/pynite_check.json`: `rigid`, `compare` |
| members, joints and lengths of both bridges | `data/episode.json`: `counts`, `geometry` |
| the six other briefs (−27 %, −14 %, −18 %, −25 %, −33 % …) | `data/candidates/candidates_table.md`, `data/results_summary.csv` |
| the collapse picture | `data/progressive.json`, `data/collapse_evolved.json`, `data/collapse_pratt.json` (the visualisation's input) |

## What is here

- `PROTOCOL.md`: the rules, frozen before any run.
- `code/`: everything in Python with numpy.
  - The solver and the evolutionary search, in `ndt_engine.py`.
  - The briefs, in `briefs.py`.
  - The runner, the analysis and the PyNite cross-check.
  - The break-test visualisation, in `collapse_xpbd.py`.
- `data/candidates/`: every run, for 7 briefs × 2 seeds. Each one has its winner, its textbook design, its analysis, its history and its full lineage (`*_lineage.npz`).
- `data/run_bridge40_s1/`: the episode's run. It holds every ancestor of the winner, the family tree, and generation 0, all 160 starting designs.
- `data/results_summary.csv`: one row per run, giving the steel mass of the evolved and textbook designs, the saving, the designs evaluated and the run time.
- `seeds.json`.

## Licence (proposed; to be confirmed at publication)

- Code: MIT (`LICENSE`).
- Data and results: CC BY 4.0 (`LICENSE-DATA.md`).

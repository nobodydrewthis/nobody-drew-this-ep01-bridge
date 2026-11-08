# Episode 1 test protocol (frozen 6 Nov 2026, 14:16, before any full run)

This file is written before the candidate runs and is not edited after them. If a bug is found later, the fix,
the reason and a full re-run of every candidate are logged in `../NOTES.md`; the rules below never move to make
evolution win.

## Physics (the same for every design, evolved or textbook)
- 2D pin-jointed truss, linear elastic, small displacements. Engine: `ndt_engine.py`.
- Steel S355: E 210 GPa, fy 355 MPa, 7,850 kg/m3. Members are circular hollow sections with D/t = 25, any area
  above a minimum (bridges 10 cm2, cranes 4 cm2, canopy 3 cm2).
- Compression: Eurocode 3 buckling curve "a" (imperfection factor 0.21) on the full member length (pinned ends),
  not bare Euler. Tension: yield.
- Slenderness L/r at most 200 in compression and 300 in tension (the crane may use pendant rods in tension, with no
  tension limit; the textbook jib uses them too).
- Global stability: the stiffness matrix must be positive definite (no mechanism, no near-mechanism pivot below 1e-9).
- Loads: factored 1.35 dead and self-weight, 1.5 live (rail 1.45; wind uplift cases take dead at 1.0). Self-weight of
  every member and every joint is included.
- Joints: each joint weighs 12% of the textbook design's member steel divided by its number of joints (fixed per brief
  before evolution). Gusset plates and bolts are real steel; this charges every design the same price per joint.
- Sizing: fully stressed design (every member as small as its worst load case allows), then a strict check that every
  member is at or below 100% in every design case. The same code sizes the textbook design.
- Geometry rules: members between the brief's minimum and maximum length; a member that passes a joint closer than the
  clearance is split at that joint; no crossing members (where two would cross, the longer is removed); free joints
  need two members, and a joint whose two members are within 5 degrees of straight is dissolved; everything inside the
  brief's design space; symmetric briefs evolve half and mirror it.

## Evolution
- Mutation and selection, no crossover: every design has exactly one parent, so the family tree is real.
- Population 160, 400 generations, tournament of 3, the best 4 survive unchanged. Seeds 1 and 2 for every brief.
- Generation 0: random clouds of 4-24 joints, connected shortest-first without crossings.
- Mutations: move joints (heavy-tailed steps), add a member, remove a member (half the time the least used one),
  split a member with a new braced joint, add a joint tied to its 3 nearest, merge a joint into its neighbour, flip a
  shared triangle edge. Each child gets 1 + Poisson(0.6) of them.
- Fitness: total steel mass in tonnes (members + joints). Designs that fail any check are infeasible.

## Cleanup rule (disclosed, applied to the winner only)
Remove the least used member (below 4% of its capacity in every case) while the design stays stable, passes every
check, and gets lighter after re-sizing. Repeat until nothing qualifies.

## Comparison with the textbook design
1. Mass: both sized by the same rules for the same load cases. Report the difference.
2. Equal mass: scale the evolved member areas so both weigh the same; ramp the live load of every design case
   (dead load and self-weight held at 1.0) to the first member failure. The evolved design must hold more in every
   case to pass the Design gate.
3. Unseen load (not used in design): ramp to first failure and to collapse (failed members removed one by one).
4. Look: share of evolved members that coincide with a textbook member (both ends within 3% of the brief's size).
   Under 50% is required, plus the owner's eye at 168 px.

## Candidate briefs (`briefs.py`)
| Brief | Textbook opponent | Load cases |
|---|---|---|
| bridge40: 40 m road bridge, structure above the deck | Pratt truss, 5 m deep | traffic full, left half, right half; unseen: 180 t abnormal load at 12-16 m |
| rail30: 30 m railway bridge | Warren truss with verticals, 4.5 m deep | locomotive at 8 positions; unseen: derailed axle load |
| jib50: 50 m hammerhead crane jib, 9 m apex | Warren lattice jib 2 m deep + two pendants | trolley at 20 positions on the load chart (8 t to 15 m, 120 t·m); unseen: 8 t at the tip |
| jib50flat: 50 m flat-top crane jib, 4 m height limit | Warren lattice jib 2 m deep | trolley at 20 positions (6 t, 90 t·m); unseen: 6 t at the tip |
| cantilever20: 20 m cantilever from a wall (Michell's problem) | tapered N-truss, 8 m deep at the wall | 600 kN tip load; unseen: tip load tilted 30 degrees |
| canopy24: 24 m canopy on one 3 m foundation | umbrella canopy (lattice mast + tapered cantilevers) | snow full, left, right; wind uplift from either side; unseen: 12 t snow drift at one edge |
| balsa30: 30 cm balsa competition bridge | Pratt truss 60 mm deep | 20 kg at the centre |

Balsa: 160 kg/m3, E 3.5 GPa, 12 MPa compression, 20 MPa tension, square sticks from 1.6 mm, timber buckling curve
(0.2, plateau 0.3), joints assumed stronger than the sticks, no load factors (a test, not a code design).

## Selection
Pick the brief with the best story and the strangest honest winner, say on screen that it was chosen among these
seven, and publish the other results. A loss or a rediscovery is reported as one.

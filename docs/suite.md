# suite: the dataset

A `Suite` is `members, seeds, bonds, legs`; a member is a family with every knob and the sizes it is drawn at. Every
combination is one instance. `SUITES` holds two:
- `reference`, the test set: 10 members, seeds 0-2, uniform bonds, four legs families. 583 instances.
- `classifier`, the training set: closed networks only, 17 members and more sizes, seeds 1000-1009, uniform and spread
  bonds.

The two share some networks (a closed uniform lattice does not depend on the seed); a shared network has the same id
in both. A bigger dataset is more rows of a suite.

`build(suite)` runs plan -> generate each entry -> drop duplicates (same id) -> add the Einsum Benchmark networks if
asked [BSK24]. `save` writes one JSON line per instance: the recipe and the id, never arrays (`config.json_line`, the
form of every file the package writes). `load` rebuilds every instance and checks its id.

## How many open legs
The legs of scattered and region weigh a share of the network's width W, so they keep the same weight relative to
the width at every size. W = c n^alpha bits, W the width of cotengra's greedy path [GK21]: alpha from theory (tree 0;
2D lattice and geometric 1/2; 3D 2/3; random regular, small world and hyper 1), c the geometric mean of W / n^alpha
over seeds 0-2 at the member's sizes. These two numbers only set how many legs are placed.

The share is chosen: `Suite.legs` maps each legs family to its shares (reference: scattered 0.5, region 0.5 and 1.5,
closed and mps none), one instance per share. Then k = max(1, round(share c n^alpha)), at most n for a region; closed
has k = 0, mps k = n. A member without alpha and c cannot get scattered or region legs: `plan` raises a ValueError.

## Reproducibility
`<name>.meta.json`, beside the data, stores what was asked (members with knobs and sizes, seeds, bond rules, legs with
their shares, chi, the benchmark) and its id, the flags of script 01, the date and the library versions. Script 01
reuses a file only for the same request. Other versions of networkx, numpy or cotengra may draw other networks from
the same seed; a slow test compares a fresh `build()` with the stored file.

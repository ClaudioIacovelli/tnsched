# suite: the dataset

A `Suite` is `members, seeds, bonds, recipes`; `SUITES` names them. `reference` (`MEMBERS`, `SEEDS`, `BONDS`,
`LEG_RECIPES`) is the test set; `classifier` (closed networks only, more members and sizes, seeds 1000-1009, uniform
and spread bonds) is the training set. A member is a family with every knob, and the sizes it is drawn at.
`build(suite)` runs plan -> generate each entry -> drop duplicates (same id) -> add the Einsum Benchmark
networks if asked [BSK24], and `save` / `load` write and read one JSON line per instance: the recipe and the
id, never arrays. The line is compact JSON (no spaces after `,` and `:`; `config.json_line`, the form of every
file the package writes), with the characters that `str.splitlines()` takes for a line break written as `\u0085`,
` ` and ` `, so that no editor cuts a record there. `load` rebuilds every instance and checks its id. With seeds 0-2 and uniform bonds the
reference suite holds 583 instances; a bigger dataset is more rows of a suite. The two suites share some
networks (a closed uniform lattice does not depend on the seed): they have the same id.

## How many open legs
The legs of scattered and region weigh a share of the network's width, W = c n^alpha bits, so
they keep the same weight relative to the width at every size. alpha comes from theory (tree 0;
2D lattice and geometric 1/2; 3D 2/3; random regular, small world and hyper 1); c is the
geometric mean of W / n^alpha over seeds 0-2 at the member's sizes, with W the width of
cotengra's greedy path [GK21]. These numbers only set how many legs a recipe places.

## Reproducibility
`<name>.meta.json` stores, next to the data, what was asked (the members with their knobs and sizes, the seeds, bond
rules and recipes, chi, the benchmark) and its id, the flags of script 01, the date and the library versions
(`config.library_versions`). Script 01 reuses a file only for the same request. Other versions of networkx, numpy or
cotengra may draw other networks from the same seed; a slow test compares a fresh `build()` with the stored file.

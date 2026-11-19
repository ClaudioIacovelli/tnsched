# instances: one network

A tensor network is the triple (inputs, output, size_dict): one tuple of index labels per
tensor, the open labels, and the dimension of every label. An `Instance` is the triple plus its
recipe (family, n, seed, bond rule, legs, knobs), from which it is rebuilt bit for bit; its
id is a hash of the triple (`network_id`). Only structure is stored, never arrays.

## Graphs and networks
- `tn_to_graph`: the lossless multigraph: one vertex per tensor, one per index shared by three or
  more tensors (a hyperedge), one per open leg; a bond between two tensors is an edge.
- `simplify`: merges parallel bonds (their weights, log2 of the dimensions, add) and drops traces.
- `remove_legs`: drops the open legs. `Instance.core = remove_legs(simplify(graph))` is the
  closed simple graph H on which the descriptors are measured.
- `graph_to_tn`: a graph turned into a closed network, one index of dimension chi (default
  `instances.CHI` = 2) per edge. For example, `graph_to_tn(nx.grid_2d_graph(2, 2))` is the closed 2x2 lattice: 4 tensors, 4 bonds of dimension 2.

## Families
| family | knobs | graph | width grows like |
|---|---|---|---|
| lattice | dim | `nx.grid_graph`, side n^(1/dim) | n^(1/2) for dim 2, n^(2/3) for dim 3 |
| randreg | degree | `nx.random_regular_graph` | n |
| geometric | dim, r_scale | `nx.random_geometric_graph`, radius r_scale / n^(1/dim) | as the lattice |
| smallworld | degree, p | `nx.newman_watts_strogatz_graph`: a ring plus random shortcuts | n |
| tree | - | `nx.random_labeled_tree` | constant |
| hyper | degree, hyper_share, rank | a regular graph plus round(hyper_share n) hyperedges, each on `rank` tensors | n |

`instances.KNOBS` lists the knobs of each family, and `generate` needs every one of them (a missing
knob is a `ValueError` that names it): no number is chosen here, `suite.py` says which values make
the dataset. `params["knobs"]` stores all of them, in the order of `KNOBS`, so an instance says what
it was built with, and `Instance.name` and `Instance.member` print them all. Every family checks its
own arguments (`CHECKS`): a size it cannot have raises a `ValueError` that says why.

The Einsum Benchmark [BSK24] (168 networks in 7 categories, gm, lm, mc, qc, rnd, str, wmc) is not
a family: `benchmark.load` takes its networks as they come, and they may be disconnected
(`Instance.components`).

## Connectivity
Every generated instance is connected: a disconnected network is its pieces contracted
independently, which teaches nothing new. Lattices, trees and the small-world ring are connected
by construction (`params["connected_by"] == "construction"`); the other families are redrawn
until connected (`"conditioning"`), which is an exact draw from the family conditioned on
connectivity. `params["attempt"]` says which draw was kept.

## Bonds and legs
Bond dimensions are chi on every bond ("uniform") or 2, 4, 8 or 16 per index ("spread", as in
the Einsum Benchmark); the graph does not change. Open legs follow a pattern: closed (none, so
the network is a number), scattered (on random tensors), region (on a connected block, as for a
reduced density matrix), mps (one per tensor). Each leg is a new index of dimension `d_open`
(default chi). `generate` runs no path search: it stores `log2_output` (the output tensor) and
`log2_total`, the sum of log2 d over all indices: no intermediate of any contraction path has more
than 2^log2_total entries.

## Seeds
One stored seed gives three independent random streams, `SeedSequence([seed, stream, attempt])`:
0 the graph, 1 the bond dimensions, 2 the legs. With the
same seed, instances that differ only in their legs or bond rule share the same graph: each random choice
has its own stream.

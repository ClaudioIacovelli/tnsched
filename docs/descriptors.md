# descriptors: the structure of a network, in numbers

`split(inst.core, seed)` cuts the core H of a network into connected components; `measure(component)` gives one row
of numbers for each, keyed by the component's id. Each number is exact, a bound (lower or upper) or an estimate of a
quantity that bounds the contraction cost; the table at the end says which, on which graph, from which source and at
what cost. Above its size cap a row is None and its name goes to the row's "skipped" list. A cap is counted in the
unit that predicts the row's cost (tensors, indices, pairs or edges), where one call takes about 2 s.

## From an instance to what is measured
- `inst.graph`: the whole network, lossless: one vertex per tensor, an edge per bond, one vertex per hyperedge (an
  index on 3 or more tensors) and per open leg; parallel bonds and traces are kept.
- `inst.core` = H: the same network simplified (parallel bonds merged, their log2 dimensions added; traces dropped)
  and closed (open legs removed). With bonds only, H is tensors and bonds; a hyperedge stays a vertex joined to its
  tensors.
- a connected component: a connected piece of H, from `split`. Every row is measured on one component; on this page
  H means the component's.

## The views of a component
- H itself: treewidth, planarity and curvature are measured here.
- the incidence graph: tensors and indices as vertices, an edge between a tensor and each of its indices (with bonds
  only, H with every edge subdivided). Distances between tensors are measured here: one index is two steps, a bond and
  a hyperedge alike (on H a bond is one step and a hyperedge two).
- the index graph L: one vertex per index, weighted by log2 of its dimension, an edge when two indices share a tensor
  (with bonds only, the line graph of H).
- the cut graph: tensors only, every index of rank r a clique on its tensors (the primal graph), weighted
  w / (floor(r/2) ceil(r/2)) per pair; with bonds only, H. The cut rows are measured here.

The cut w(dS) of a set S of tensors is the sum of log2 d over the indices with tensors on both sides, each once. The
width W of a contraction tree is its largest cut; b is the smallest balanced cut. The cut is not a graph. A split
k | r - k of an index cuts at most floor(r/2) ceil(r/2) pairs of its clique, so no cut of the cut graph exceeds the
true one: its spectrum gives valid lower bounds, and its bisections are measured on the true cut. The primal graph
serves only for cuts: for treewidth it would turn one index on n tensors into a clique of treewidth n - 1.

T* is the log2 cost of the largest step (one pairwise contraction) of the best contraction tree. tw_w(L) is the
smallest, over the tree decompositions of L, of the heaviest bag, a bag weighing the sum of the log2 dimensions of its
indices (no -1). [MS08] prove T* = tw_w(L) (with every dimension 2: tw(L) + 1), so bounds on tw_w(L) are bounds on
T*; the cost C of the best tree obeys 2^T* <= C <= (n - 1) 2^T*.

## The brackets
- **treewidth** of H: `tw_lower` = max(degeneracy, minor-min-width [BKW06], spectral bound [GHN24]); `tw_upper` = the
  width of a min-fill elimination order (any order is a tree decomposition).
- **weighted treewidth** T* = tw_w(L): `tw_w_lower` = the max of the heaviest tensor, min w times a bag of tw(L) + 1
  indices (degeneracy and minor-min-width of L), and min w times `tw_lower`; `tw_w_upper` = the heaviest min-fill bag
  of L (an elimination order of L is a contraction order).
- **width**, through the balanced cut b: `b_lower` = 2 lambda_2 n / 9 on the cut graph (a cut with n/3 to 2n/3
  tensors on a side is at least lambda_2 |S| |V \ S| / n [Moh92], so it also bounds W); `b_upper` = the cut of
  `halve()`: the best of six balanced splits refined by Kernighan-Lin (one from the Fiedler vector of the cut graph,
  five random), measured on the true cut.

## The estimates
- **separator exponent**: recursive halving gives b(m) for pieces of m tensors; `alpha_bisection` fits
  b(m) = a + scale m^alpha and `alpha_bisection_err` is its standard error (alpha = 0 for trees, 1/2 in 2D, 2/3 in
  3D, 1 for expanders).
- **dimension**, from the balls: |B(r)| around up to 8 sources grows like r^D (a dimension) or e^(c r) (no finite
  dimension); the majority of the sources decides. Three rows: the flag "exponential", D, c (one of the two None).
- **dimension**, from the walk: the return probability of a random walk on the tensors [ZHS06] decays to its
  stationary value like k^(-d_s/2) (spectral dimension d_s) or e^(-gamma k) (a gap); the same three rows.

## The other rows
- **size**: tensors, indices, incidences, and the mean and spread of log2 d.
- **degree**: the mean and largest degree, the hyperedges, their share and the largest rank.
- **distance**: the diameter, a lower bound on it [MLH09] and the girth, on the incidence graph.
- **expansion**: lambda_2 of H.
- **topology**: planarity and a lower bound on the genus.
- **curvature**: the Forman curvature of the edges of H [For03], its mean and its minimum.

## The file
Script 02 appends one row per component to `results/descriptors/<name>.jsonl`, so it can be stopped and resumed.
Its meta (`<name>.meta.json`) stores the table id (a hash of the rows, caps and units of the probes that ran;
`--fast` runs only the fast ones, another table), the seed of `split` (it fixes the sources of the balls and the walk
and the starts of the bisections), the flags of the script, the date and the library versions. `load` refuses a file
of another table or seed: run 02 with `--fresh`. Beside it, `<name>_components.jsonl` holds one line per instance:
its id and the id and number of tensors of each component, largest first. `features.samples` reads it and builds no
graph.

## Known limits
- Random trees: their spectral dimension is 4/3 in the limit, but the walk rows read 1.29 to 1.46 for single trees
  of 256 to 4096 tensors (seeds 0-2): the local slope of the return probability changes along the window from one
  tree to the next.
- Small worlds (degree 4, p = 0.05 and 0.1, at 256 and 1024 tensors): the balls grow exponentially while the walk
  fits a power law with d_s between 1.9 and 2.6. The walk is crossing over from the ring to an expander, so that d_s
  is not a dimension: the balls count volume up to half the eccentricity, the walk measures diffusion until it mixes.
- `alpha_bisection` is fitted between 0.01 and 1. At 1 it is the expander value (a separator never grows faster than
  m); at 0.01, with no error, b(m) did not grow over the measured sizes, as for random geometric graphs in 2D at 256
  tensors, whose dense clusters cost as much to cut as the whole network. `alpha_bisection_err` is None when the
  profile has three points or the fit has no finite covariance; with fewer than three distinct sizes both rows are
  None.

## The table
| rows | kind | bounds | on | source | complexity | cost |
|---|---|---|---|---|---|---|
| `n_tensors`, `n_indices`, `n_incidences`, `log2_d_mean`, `log2_d_std` | exact | - | - | - | O(m) | fast |
| `degree_mean` | exact | - | - | - | O(m) | fast |
| `degree_max` | exact | - | - | - | O(m) | fast |
| `n_hyperedges` | exact | - | - | - | O(m) | fast |
| `hyperedge_fraction` | exact | - | - | - | O(m) | fast |
| `hyperedge_size_max` | exact | - | - | - | O(m) | fast |
| `diameter` | exact | - | incidence | - | O(n m) | slow, <= 1,500 tensors |
| `diameter_lower` | lower | - | incidence | MLH09 | O(m) | fast |
| `girth` | exact | - | incidence | - | O((n + m) m) | slow, <= 1,000 tensors |
| `ball_exponential`, `ball_dimension`, `ball_growth_rate` | estimate | D | incidence | - | O(m) | fast, <= 200,000 tensors |
| `walk_exponential`, `walk_dimension`, `walk_decay_rate` | estimate | D | incidence | ZHS06 | O(K s sum r_e^2), K <= WALK_MAX_STEPS, s <= 32 | fast, <= 10,000 tensors |
| `tw_lower` | lower | tw | H | BKW06, GHN24 | O(m log n) (degeneracy, minor-min-width) + the spectral term up to SPECTRAL_MAX_N | fast, <= 150,000 tensors |
| `tw_upper` | upper | tw | H | - | O(n^2 log n) to O(n^4) worst | slow, <= 800 tensors |
| `tw_w_lower` | lower | T | index graph | BKW06, GHN24 | tw_lower on H + minor-min-width of the index graph, about 6 tensors per index in the largest networks | fast, <= 25,000 indices |
| `tw_w_upper` | upper | T | index graph | MS08 | O(I^2 log I) to O(I^4) worst | slow, <= 1,200 indices |
| `lambda_2` | exact | - | H | - | O(n^3) dense (n <= 300) / iterative | fast, <= 4,096 tensors |
| `lambda_2_cut` | exact | - | cut | - | O(n^3) dense (n <= 300) / iterative | fast, <= 50,000 pairs |
| `b_lower` | lower | b | cut | Moh92 | O(n^3) dense (n <= 300) / iterative | fast, <= 50,000 pairs |
| `b_upper` | upper | b | cut | - | 6 starts x <= 10 KL passes of O(m log n), + the Fiedler vector of lambda_2_cut | slow, <= 25,000 pairs |
| `alpha_bisection`, `alpha_bisection_err` | estimate | alpha | cut | - | O(m log^2 n) (halve on every piece of every level) + their Fiedler vectors | slow, <= 3,500 tensors + pairs |
| `planar` | exact | - | H | - | O(n log Delta) | fast, <= 80,000 tensors |
| `genus_lower` | lower | - | incidence | - | O((n + m) m) | slow, <= 1,000 tensors |
| `forman_mean` | exact | - | H | For03 | O(m Delta) | fast, <= 1,000,000 edges |
| `forman_min` | exact | - | H | For03 | O(m Delta) | fast, <= 1,000,000 edges |

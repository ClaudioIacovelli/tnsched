# features: the input of the model

`component_input(c)` turns one connected component into what a graph neural network reads: one row of numbers per
tensor (`x_tensor`), one per index (`x_index`), and the incidences between them (`edge`, shape (2, E): column k is
one incidence, the row of its tensor in `x_tensor` and the row of its index in `x_index`). It reads the core H, like
the descriptors: open legs and traces are not in the input, and a hyperedge is one index node whatever its rank.

## Columns
| node | column | value | divided by |
|---|---|---|---|
| tensor | `degree` | how many indices it carries | 8 |
| tensor | `log2_size` | log2 of its number of entries (the sum of log2 d of its indices) | 32 |
| index | `w` | log2 of its dimension | 4 |
| index | `log2_rank` | log2 of how many tensors it joins | 8 |

The scales keep every column of order one on the suites.

## The random-walk encoding
`rwpe` appends to every tensor and every index the probability that the walk of the descriptors (`descriptors.walk`
[ZHS06]) is back where it started after each number of steps in `RWPE_STEPS` [DLL22]: every even step up to 16, then
about geometric up to 128 (on the incidence graph a walk comes back only after an even number of steps). The value
stored is log(p) / `RWPE_SCALE`, with p floored at `RWPE_FLOOR`. The powers of the walk are dense matrices, so a
component above `RWPE_MAX_TENSORS` = 2048 tensors raises a `ValueError` and script 03 skips it.

## The ball encoding
`balls` appends to every tensor and every index |B_r|, the number of tensors reached from it crossing at most r
indices, for r in `BALL_RADII` = 1, 2, 3, 4 (r = 4 is 8 steps of the incidence graph). An index crosses itself first,
so at r = 1 it counts its rank. The value stored is log2 |B_r| / `BALL_SCALE` (1 at 1024 tensors).

Why: a message-passing network that sums its messages is bounded by the 1-WL test. It counts walks, not distinct
nodes, so two graphs with the same unfolding tree get the same states even where one has short cycles. |B_r| counts
distinct tensors: a short cycle makes the ball smaller than on a tree. It is an exact count, equal on nodes a
symmetry of the graph exchanges.

How: one sparse boolean product per radius, every node at once: R_r = R_(r-1) (1 + M), with M = B B^T the tensors
that share an index. 0.16 s on the largest classifier component (1024 tensors, 9509 indices). (`ball_fit` counts the
same balls by BFS from 8 sources, to fit one dimension per component.)

Not used: Laplacian eigenvectors, because their sign, and on lattices the basis of a degenerate eigenspace, are not
functions of the graph. Not in the input: open legs. The input, its key (the component id) and its labels are those
of the core, as for the descriptors; legs enter the input of the scheduler, not this one.

## Which columns a model reads
`ENCODINGS` = {"rwpe": 14, "balls": 4} lists the encodings with their widths, in the order of their columns after
the native ones. `columns(kind, encodings)` gives the columns of `x_tensor` or `x_index` that a model with these
encodings reads: the native ones always, then each encoding it asks for.

## The file
Script 03 writes `results/features/<name>.npz`: three arrays per component, `<id>/x_tensor`, `<id>/x_index` and
`<id>/edge`, with the component's id as in the descriptors file. `features.load` reads it without pickle and returns
{id: ComponentInput}. Its meta (`<name>.meta.json`) stores the features id (a hash of the columns, scales, steps and
caps above), the flags of the script, the date and the library versions; `load` refuses a file made with another
table: run 03 with `--fresh`. The arrays are plain numpy (float32, int64 for `edge`): any framework reads them, and
`torch.from_numpy` makes tensors of them without a copy.

## From the files to the model
1. `split_labels(records, how, held_out)` labels every record of the instances file (`config.load_lines`: the JSON
   lines, no network built) "train", "val" or "test": at random by a hash of its graph, with a family held out, or by
   size. Instances that differ only in their legs or bond rule have one graph, so one label.
2. `samples(records, components_by_instance, descriptors_by_component, features_by_component, target_names)` gives one
   `(ComponentInput, y)` for each component of those records. The components come from the components file of script
   02 (by instance id); the input from the features file of 03 and the labels from the descriptors file of 02 (both
   by component id). y holds the descriptors named in `target_names`, in that order, NaN where missing.
3. `unseen(pairs, train_pairs)` drops from val and test every component that is also in train: two graphs can share a
   component (two seeds of a small tree, a small piece of two disconnected networks).

Which rows are labels: the size group is given (`GIVEN`), `alpha_bisection` and its error are left out (`LEFT_OUT`),
every other row is predicted. `TARGETS` maps each to how it is learnt: "binary" (a probability), "linear"
(standardised as it is) or "log" (log1p, then standardised). `SLOW_TARGETS` are those whose probe is slow.

This module does not batch. A model that batches B graphs (PyTorch Geometric does it this way) concatenates their
rows, since they have different numbers of tensors and indices, shifts `edge` by the rows of the graphs before it, so
no edge joins two graphs, and records which graph each tensor row belongs to; y becomes (B, K).

# features: the input of the model

`component_input(c)` turns one connected component into what a graph neural network reads: one row of numbers
per tensor (`x_tensor`), one per index (`x_index`), and the incidences between them (`edge`: row 0 the position
of the tensor in `c.tensors`, row 1 the position of the index in `c.indices`). It reads the closed core H, like
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
`rwpe` appends to every tensor and every index the probability that the walk of the descriptors
(`descriptors.walk` [ZHS06]) is back where it started after each number of steps in `RWPE_STEPS` [DLL22]: every
even step up to 16, then about geometric up to 128 (a walk on the incidence graph comes back only after an even
number of steps). The value stored is log(p) / `LOG_SCALE`, with p floored at `LOG_FLOOR`. The powers of the
walk are dense matrices, so a component above `MAX_DENSE` = 2048 tensors raises a `ValueError` and script 03
skips it.

## The file
Script 03 writes `results/features/<name>.npz`: three arrays per component, under `<id>/x_tensor`,
`<id>/x_index` and `<id>/edge`, with the component's id as in the descriptors file. `features.load` reads it
without pickle and returns {id: ComponentInput}. Its meta (`<name>.meta.json`) stores the features id (a hash of
the columns, scales, steps and caps above), the flags of the script, the date and the library versions; `load`
refuses a file made with another table: run 03 with `--fresh`. The arrays are plain numpy (float32, and int64
for `edge`): any framework reads them, and `torch.from_numpy` makes tensors from them without a copy.

# tnsched

Learning to contract tensor networks: a dataset of networks of known structure, the numbers that describe that
structure, and the input a graph neural network reads from each network.

## What it is for
The cost of contracting a tensor network depends on the order in which its tensors are contracted, and the best
order is NP-hard to find [XZL23]. The cost of the best order is set by the structure of the network: its treewidth,
how well it splits into balanced halves, how fast it grows [MS08, GK21]. tnsched builds networks whose structure is
known by construction (lattices, random regular graphs, random geometric graphs, small worlds, trees, hypergraph
networks) and measures, for each, bounds and estimates of these quantities on the graph alone.

## Scope
- `instances`: one network, rebuilt bit for bit from its recipe (family and its knobs, size, seed, bond dimensions,
  open legs). Only structure is stored, never arrays.
- `suite`: which networks make the dataset, and the file that stores them.
- `descriptors`: the structural numbers of a network, each a bound or an estimate of a quantity that bounds the
  contraction cost, with its cost of computation and the theorem behind it.
- `features`: the input of a graph neural network for each component, as numpy arrays: a few numbers per tensor and
  per index, the return probabilities of a random walk [DLL22], and the sizes of the balls around each node.
- `benchmark`: a loader for the Einsum Benchmark [BSK24], real-world networks taken as they come.

These are the inputs of a learned contraction scheduler: a graph neural network that reads the features and is
trained to predict a contraction order, with the descriptors as auxiliary targets. The scheduler is not part of this
release.

## What is new
Auxiliary targets for graph neural networks are known [Car97, CCB20], and so are learned contraction orders
[MMM22, GHS22, KSO20]. What we did not find is a model that learns the contraction order together with descriptors
that bound its cost (the weighted treewidth of the index graph, which equals the largest step of the best
contraction tree [MS08]; the balanced cut and its scaling exponent; the growth dimension), measured on hypergraph
networks across classes and sizes. This release provides the dataset and the descriptors that such a model needs.

## Install
Python 3.11 or later.

    python -m pip install -e .

installs the package (numpy, scipy, networkx and cotengra) in editable mode; the scripts import it, so they run
from any folder. `python -m pip install -e ".[test]"` adds pytest for the tests. The Einsum Benchmark (optional,
552 MB, 168 networks) is at [doi:10.5281/zenodo.11477304](https://doi.org/10.5281/zenodo.11477304):

    python scripts/00_get_benchmark.py

downloads `instances.zip`, checks its md5 and unzips it into `data/einsum_benchmark/`. `--zip PATH` uses a copy
already on disk; the environment variable `TNSCHED_BENCHMARK` reads the instances from another folder.

## Run

    python -m pytest                           # -m slow runs the slow tests instead (the 168 benchmark files, the stored reference)
    python scripts/01_build_instances.py --build reference   # results/instances/reference.jsonl is in the repository: checked and used
    python scripts/02_compute_descriptors.py --suite results/instances/reference.jsonl
                                                             # -> results/descriptors/reference.jsonl and reference_class_table.md
    python scripts/03_compute_features.py --suite results/instances/reference.jsonl
                                                             # -> results/features/reference.npz and reference.meta.json

- `01 --build` takes a suite of `suite.SUITES` (`reference`, `classifier`) or `manual`; `--benchmark` adds the 168
  benchmark networks. Every file's meta stores what was asked, and a file is reused only for the same request. A
  manual build, or a suite changed by `--max-n`, `--seeds` or `--bonds`, gets the date and time in its name
  (`manual_20261118-101500.jsonl`); `--tag NAME` names the file instead (`<tag>_<suite>.jsonl`).
- `02` measures every distinct component of an instances file into a descriptors file named after it (one row per
  component id); it can be stopped and resumed.
- `03` writes the features of every component.
- 02 and 03 pick the instances file with `--suite PATH`, with `--build`, or by asking.

A manual build gives the family and all its knobs as JSON (in Windows PowerShell 5.1 the inner quotes need a
backslash):

    python scripts/01_build_instances.py --build manual --family lattice --knobs '{"dim": 2}' --sizes 16 64 --seeds 0
    python scripts/01_build_instances.py --build manual --family lattice --knobs '{\"dim\": 2}' --sizes 16 64 --seeds 0   # Windows PowerShell 5.1

From Python:

    from tnsched import descriptors, features, instances, suite

    inst = instances.generate("lattice", 64, seed=0, dim=2)                # an Instance, every knob given
    rho = instances.generate("lattice", 64, seed=0, dim=2, legs="region", k=4)   # with 4 open legs
    piece = descriptors.split(inst.core, 0)[0]                             # its one connected component
    row = descriptors.measure(piece)                                       # the descriptors of that component
    x = features.component_input(piece)                                    # its input to the model: numpy arrays
    insts = suite.load("results/instances/reference.jsonl")                # the stored dataset

## Layout

    src/tnsched/instances.py     one network: families, bond rules, open legs, the Instance record
    src/tnsched/suite.py         Suite, SUITES (reference, classifier); build, save, load
    src/tnsched/descriptors.py   the table of descriptors and measure()
    src/tnsched/features.py      the input of the model: NATIVE_FEATURES, rwpe, ball_sizes, component_input, split_labels, samples, unseen
    src/tnsched/benchmark.py     the Einsum Benchmark loader
    src/tnsched/config.py        paths, the JSON-lines format, the meta file beside every file
    scripts/                     00 get the benchmark, 01 build a suite, 02 measure the descriptors, 03 compute the features; _shared.py is what 01, 02 and 03 share
    tests/                       one test file per module, and test_scripts.py for scripts/
    docs/                        one page per module: instances.md (with the benchmark), suite.md, descriptors.md, features.md
    results/instances/           reference.jsonl and classifier.jsonl, each with its meta (the stored suites and what built them)
    results/descriptors/         the descriptors of both suites (.jsonl, .meta.json, _class_table.md, _components.jsonl); the features are made by 03, not stored

## Authorship
Claudio Iacovelli designed tnsched (the modules in src/, the scripts and the tests: what each one computes and how the
code is laid out), decided every specification and reviewed the work; some specifications were drafted with Claude
and decided by him. The code was written by Claude (Anthropic) from those specifications. Claudio reviewed every
line of src/; he reviewed the scripts for their logic (what each function does, the flags, the files they
read and write, the messages), not every call to the standard library line by line. The tests were also written by Claude and reviewed as a list of the properties they check,
one line per test, not line by line. The text of this README and of docs/ was drafted by Claude and edited by Claudio.

## References
[BKW06] H. L. Bodlaender, T. Wolle and A. M. C. A. Koster, Contraction and treewidth lower bounds, J. Graph Algorithms Appl. 10, 5-49 (2006), doi:10.7155/jgaa.00117.
[BSK24] M. Blacher, C. Staudt, J. Klaus, M. Wenig, N. Merk, A. Breuer, M. Engel, S. Laue and J. Giesen, Einsum Benchmark: enabling the development of next-generation tensor execution engines, NeurIPS 2024 Datasets and Benchmarks Track, doi:10.52202/079017-3111; data at https://doi.org/10.5281/zenodo.11477304.
[Car97] R. Caruana, Multitask learning, Machine Learning 28, 41-75 (1997), doi:10.1023/A:1007379606734.
[CCB20] G. Corso, L. Cavalleri, D. Beaini, P. Lio and P. Velickovic, Principal neighbourhood aggregation for graph nets, NeurIPS 2020.
[DLL22] V. P. Dwivedi, A. T. Luu, T. Laurent, Y. Bengio and X. Bresson, Graph neural networks with learnable structural and positional representations, ICLR 2022, arXiv:2110.07875.
[For03] R. Forman, Bochner's method for cell complexes and combinatorial Ricci curvature, Discrete Comput. Geom. 29, 323-374 (2003), doi:10.1007/s00454-002-0743-x.
[GHN24] T. Gima, T. Hanaka, K. Noro, H. Ono and Y. Otachi, An improved spectral lower bound of treewidth, arXiv:2404.08520 (2024).
[GHS22] A. Gatti, Z. Hu, T. Smidt, E. G. Ng and P. Ghysels, Graph partitioning and sparse matrix ordering using reinforcement learning and graph neural networks, J. Mach. Learn. Res. 23, no. 303 (2022).
[GK21] J. Gray and S. Kourtis, Hyper-optimized tensor network contraction, Quantum 5, 410 (2021), arXiv:2002.01935.
[KSO20] T. Khakhulin, R. Schutski and I. Oseledets, Learning elimination ordering for tree decomposition problem, NeurIPS 2020 Workshop on Learning Meets Combinatorial Algorithms (LMCA); preprint arXiv:1910.08371, titled Graph convolutional policy for solving tree decomposition via reinforcement learning heuristics.
[MLH09] C. Magnien, M. Latapy and M. Habib, Fast computation of empirically tight bounds for the diameter of massive graphs, ACM J. Exp. Algorithmics 13, art. 10 (2009).
[MMM22] E. Meirom, H. Maron, S. Mannor and G. Chechik, Optimizing tensor network contraction using reinforcement learning, ICML 2022, PMLR 162.
[Moh92] B. Mohar, Laplace eigenvalues of graphs - a survey, Discrete Math. 109, 171-183 (1992), doi:10.1016/0012-365X(92)90288-Q.
[MS08] I. L. Markov and Y. Shi, Simulating quantum computation by contracting tensor networks, SIAM J. Comput. 38, 963-981 (2008), doi:10.1137/050644756, arXiv:quant-ph/0511069.
[XZL23] J. Xu, H. Zhang, L. Liang, L. Deng, Y. Xie and G. Li, NP-hardness of tensor network contraction ordering, arXiv:2310.06140 (2023).
[ZHS06] D. Zhou, J. Huang and B. Scholkopf, Learning with hypergraphs: clustering, classification, and embedding, NIPS 19, 1601-1608 (2006).

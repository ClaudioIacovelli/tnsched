"""benchmark.py: the safe unpickler, names, load (on a small pickle made here, in the benchmark's format), the renaming of
surrogate labels, and with the Einsum Benchmark on disk (needs_benchmark) the 168 networks and their stored paths."""
import io
import pickle
from collections import Counter

import cotengra as ctg
import numpy as np
import pytest
from conftest import signature

from tnsched import benchmark, config, suite
from tnsched.benchmark import _NumpyOnlyUnpickler
from tnsched.instances import graph_to_tn, tn_to_graph

needs_benchmark = pytest.mark.skipif(not any(config.BENCHMARK.glob("*.pkl")), reason=f"no Einsum Benchmark in {config.BENCHMARK}")


def test_the_unpickler_reads_numpy_arrays_and_refuses_any_other_global():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    loaded = _NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(array, protocol=4))).load()               # protocol 4: the _reconstruct form
    assert loaded.dtype == array.dtype and (loaded == array).all()
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        _NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(Counter("ab")))).load()                        # needs collections.Counter


def test_names_are_the_sorted_pkl_stems(tmp_path):
    for stem in ("b_two", "a_one"):
        (tmp_path / f"{stem}.pkl").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    assert benchmark.names(tmp_path) == ["a_one", "b_two"]


def test_load_reads_a_pickle_in_the_benchmark_format(tmp_path):
    """(format string, tensors, two stored paths, output): the instance has the network of the string and the paths in steps of ints."""
    shapes = [(2, 3, 4), (3, 2, 4), (2, 3, 4)]                                                    # "abx,bcx,cdx->ad": x is a hyperedge, a and d open
    paths = (([(0, 1), (0, 1)], 3.0, 1.5), ([(1, 2), (0, 1)], 4.0, 2.5))
    with open(tmp_path / "qc_toy.pkl", "wb") as fh:
        pickle.dump(("abx,bcx,cdx->ad", [np.zeros(s, dtype=np.float32) for s in shapes], paths, None), fh, protocol=4)
    inst = benchmark.load("qc_toy", tmp_path)
    assert inst.family == "benchmark" and inst.seed is None and inst.name == "benchmark-qc_toy"
    assert inst.inputs == [("a", "b", "x"), ("b", "c", "x"), ("c", "d", "x")] and inst.output == ("a", "d")
    assert inst.size_dict == {"a": 2, "b": 3, "x": 4, "c": 2, "d": 3} and all(type(d) is int for d in inst.size_dict.values())
    assert inst.params["n"] == 3 and inst.params["category"] == "qc" and inst.params["relabelled"] == 0
    assert inst.params["paths"] == {"opt_size": {"path": [(0, 1), (0, 1)], "size": 3.0, "flops": 1.5},
                                    "opt_flops": {"path": [(1, 2), (0, 1)], "size": 4.0, "flops": 2.5}}
    ctg.ContractionTree.from_path(*inst.network, path=inst.params["paths"]["opt_size"]["path"])      # the stored path rebuilds as a tree


def test_a_pickle_with_fewer_than_two_stored_paths_is_refused():
    with pytest.raises(ValueError, match="2 stored paths"):
        benchmark._stored_paths((([(0, 1)], 3.0, 1.0),))


def test_surrogate_labels_are_renamed_and_nothing_else_changes():
    """A lone surrogate is not a character and cannot be written as UTF-8: it becomes an unused character above U+DFFF."""
    inputs, output = [("\ud801", "\udc00", "a"), ("\ud801", "b"), ("\udc00", "b", "a")], ("b",)
    size_dict = {"\ud801": 2, "\udc00": 3, "a": 4, "b": 5}
    ins, out, sd, k = benchmark.without_surrogates(inputs, output, size_dict)
    assert k == 2 and not any(ord(char) in benchmark.SURROGATES for label in sd for char in label)
    assert out == ("b",) and ins[0][2] == "a"                                                      # the other labels keep their names
    assert [[sd[e] for e in t] for t in ins] == [[size_dict[e] for e in t] for t in inputs]
    assert ins[0][0] == ins[1][0] and ins[0][1] == ins[2][0]                                       # the same structure
    "|".join(sd).encode("utf-8")                                                                   # now writable as UTF-8
    assert benchmark.without_surrogates(ins, out, sd) == (ins, out, sd, 0)


@needs_benchmark
def test_the_benchmark_has_168_networks_in_7_categories():
    names = benchmark.names()
    assert len(names) == 168 and {s.split("_")[0] for s in names} == {"gm", "lm", "mc", "qc", "rnd", "str", "wmc"}


@needs_benchmark
def test_a_stored_path_rebuilds_as_a_contraction_tree():
    inst = benchmark.load("qc_circuit_n49_m14_s9_e6_pEFGH_simplified")
    assert inst.family == "benchmark" and inst.seed is None
    assert ctg.utils.inputs_output_to_eq(inst.inputs, inst.output) == inst.eq
    for key in ("opt_size", "opt_flops"):
        path = inst.params["paths"][key]["path"]
        assert len(path) == len(inst.inputs) - 1
        ctg.ContractionTree.from_path(*inst.network, path=path)


@needs_benchmark
def test_benchmark_instances_survive_save_and_load(tmp_path):
    """== holds after save and load: the stored paths come back as tuples. mc_2020_017 has lone surrogates in its labels."""
    insts = [benchmark.load(name) for name in (benchmark.names()[0], "mc_2020_017")]
    suite.save(insts, tmp_path / "suite.jsonl")
    assert suite.load(tmp_path / "suite.jsonl") == insts


@pytest.mark.slow
@needs_benchmark
def test_the_size_path_and_the_flops_path_are_in_that_order():
    """path_meta[0] is the size-optimised path and path_meta[1] the flops-optimised one. Both are the best a search found,
    not optima: in 16 of the 168 one inequality fails (15 times the flops path has up to 0.09 decades more flops; in
    mc_2020_arjun_061 the flops path is better in both). So the order is tested on the counts, pinned for the md5-checked
    dataset: a swapped order would invert them."""
    smaller_size, smaller_flops = Counter(), Counter()
    for name in benchmark.names():
        p = benchmark.load(name).params["paths"]
        S, F = p["opt_size"], p["opt_flops"]
        smaller_size[(S["size"] > F["size"]) - (S["size"] < F["size"])] += 1                       # -1: S smaller
        smaller_flops[(F["flops"] > S["flops"]) - (F["flops"] < S["flops"])] += 1                  # -1: F smaller
    assert smaller_size == {-1: 107, 0: 60, 1: 1} and smaller_flops == {-1: 140, 0: 13, 1: 15}


@pytest.mark.slow
@needs_benchmark
def test_graph_to_tn_inverts_tn_to_graph_on_all_168():
    """Up to label names and axis order within a tensor: open legs and their order, traces, local indices, hyperedges, parallel bonds."""
    for name in benchmark.names():
        inst = benchmark.load(name)
        assert signature(*graph_to_tn(tn_to_graph(*inst.network))) == signature(*inst.network), name


@pytest.mark.slow
@needs_benchmark
def test_the_reference_suite_with_the_benchmark_survives_save_and_load(tmp_path):
    insts = suite.build(suite.SUITES["reference"], with_benchmark=True)
    suite.save(insts, tmp_path / "suite.jsonl")
    back = suite.load(tmp_path / "suite.jsonl")                                                    # checks every id
    assert [b.name for b in back] == [a.name for a in insts]
    relabelled = {i.params["name"] for i in insts if i.family == "benchmark" and i.params["relabelled"]}
    assert relabelled == {"mc_2020_017", "mc_2020_062", "mc_2020_082", "mc_2022_167", "wmc_2023_141"}

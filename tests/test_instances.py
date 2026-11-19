"""instances.py: graph <-> network, the families and their knobs, the streams and the draw, the bonds, the legs, the
sizes, the checks, ids and Instance. Known answers on small networks first, then the invariants, then the errors."""
from collections import Counter

import cotengra as ctg
import networkx as nx
import numpy as np
import pytest
from conftest import gen, signature

from tnsched import instances
from tnsched.instances import FAMILIES, HYPER, Instance, draw_connected, graph_to_tn, remove_legs, simplify, tn_to_graph

RANDOM_FAMILIES = ["randreg", "geometric", "smallworld", "tree", HYPER]


# --- graphs and networks: graph_to_tn, tn_to_graph, simplify, remove_legs ------------------------------

def test_the_2x2_grid_is_four_tensors_on_four_bonds():
    """graph_to_tn(grid_2d_graph(2, 2)) is the example of docs/instances.md: four tensors, four bonds of dimension chi."""
    inputs, output, size_dict = graph_to_tn(nx.grid_2d_graph(2, 2))
    assert inputs == [("a", "b"), ("b", "c"), ("a", "d"), ("c", "d")]
    assert output == ()
    assert size_dict == {"a": 2, "b": 2, "c": 2, "d": 2}


def test_tn_to_graph_has_a_vertex_per_tensor_per_hyperedge_and_per_open_leg():
    """A bond is an edge, a trace a self-loop, parallel bonds two edges; an index on three or more tensors, or open, a vertex."""
    G = tn_to_graph([("a", "a", "b"), ("b",)], (), {"a": 2, "b": 4})                              # a trace
    assert set(G.nodes) == {0, 1} and nx.number_of_selfloops(G) == 1
    assert sorted(G.edges(keys=True, data="weight")) == [(0, 0, "a", 1.0), (0, 1, "b", 2.0)]
    G = tn_to_graph([("a", "b"), ("a", "b")], (), {"a": 2, "b": 8})                               # parallel bonds
    assert sorted(G.edges(keys=True, data="weight")) == [(0, 1, "a", 1.0), (0, 1, "b", 3.0)]
    inputs, output = [("h", "l", "o"), ("h", "x"), ("h", "x")], ("o",)                            # hyperedge, open, local
    G = tn_to_graph(inputs, output, {"h": 2, "l": 8, "o": 2, "x": 2})
    assert G.nodes[("index", "h")] == {"open": False} and G.nodes[("index", "o")] == {"open": True, "pos": 0}
    assert set(G[("index", "h")]) == {0, 1, 2}                                                    # joined to every carrier
    assert G.nodes[0]["local"] == ["l"] and G.nodes[0]["local_weight"] == [3.0]                  # an index on one tensor: no edge
    assert not any(k == "l" for *_, k in G.edges(keys=True))


def test_simplify_adds_parallel_weights_drops_traces_and_counts_a_repeated_slot_once():
    """w = log2 d adds over parallel bonds; a trace joins nothing; T[h, h, x] writes h twice but h is one leg of dimension d."""
    G = tn_to_graph([("a", "b", "t", "t"), ("a", "b", "h"), ("h",), ("h",)], (), {"a": 2, "b": 8, "t": 5, "h": 4})
    H = simplify(G)
    assert type(H) is nx.Graph and nx.number_of_selfloops(H) == 0 and set(H.nodes) == set(G.nodes)
    assert H[0][1]["weight"] == 1.0 + 3.0 and H.nodes[("index", "h")] == {"open": False}          # a and b act as one index of size 16
    assert all(w > 0 for *_, w in H.edges(data="weight"))
    G = tn_to_graph([("h", "h", "x"), ("h", "y"), ("h", "z")], (), {"h": 4, "x": 2, "y": 2, "z": 2})
    assert G.number_of_edges(0, ("index", "h")) == 2                                              # the lossless graph keeps both
    H = simplify(G)
    assert H[0][("index", "h")]["weight"] == 2.0 and H[1][("index", "h")]["weight"] == H[2][("index", "h")]["weight"] == 2.0
    G = tn_to_graph([("h", "h", "x"), ("x",)], ("h",), {"h": 8, "x": 2})                          # an open leg written twice
    assert simplify(G)[0][("index", "h")]["weight"] == 3.0
    G = tn_to_graph([("a", "b", "t", "t"), ("a", "b")], (), {"a": 2, "b": 4, "t": 8})
    assert simplify(G)[0][1]["weight"] == 1.0 + 2.0


def test_remove_legs_drops_only_the_pendant_open_indices():
    """An open index on one tensor goes; one on two tensors stays; every tensor stays."""
    G = simplify(tn_to_graph([("a", "o", "p"), ("a", "o")], ("o", "p"), {"a": 2, "o": 2, "p": 2}))
    H = remove_legs(G)
    assert set(H.nodes) == {0, 1, ("index", "o")} and H.nodes[("index", "o")]["open"] is True
    assert ("index", "p") in G and H is not G                                                     # G itself is not changed
    plain, legged = gen(HYPER, 16, 0), gen(HYPER, 16, 0, legs="scattered", k=5)
    assert nx.utils.graphs_equal(plain.core, legged.core)


def test_graph_to_tn_reads_parallel_edges_loops_weights_and_index_vertices():
    """Two parallel edges are two indices, a loop is one index written twice, w is log2 d, an open leg keeps its axis."""
    inputs, output, size_dict = graph_to_tn(nx.MultiGraph([(0, 1), (0, 1), (1, 1)]), chi=3)
    assert Counter(map(len, inputs)) == Counter({2: 1, 4: 1})
    assert inputs[0] == ("a", "b") and sorted(inputs[1]) == ["a", "b", "c", "c"]
    assert output == () and size_dict == {"a": 3, "b": 3, "c": 3}
    G = tn_to_graph([("h", "l"), ("h", "x"), ("h", "x")], (), {"h": 2, "l": 8, "x": 4})
    inputs, output, size_dict = graph_to_tn(G)
    assert [sorted(Counter(t).values()) for t in inputs] == [[1, 1], [1, 1], [1, 1]]
    assert sorted(size_dict.values()) == [2, 4, 8]                                                # sizes from the weights
    assert Counter(Counter(e for t in inputs for e in t).values()) == {3: 1, 2: 1, 1: 1}          # h, x, l
    inputs, output, size_dict = [("a", "p", "q"), ("a", "r")], ("r", "q", "p"), dict.fromkeys("apqr", 2)
    G = tn_to_graph(inputs, output, size_dict)
    assert [G.nodes[("index", e)]["pos"] for e in output] == [0, 1, 2]
    back = graph_to_tn(G)
    assert len(back[1]) == 3 and signature(*back) == signature(inputs, output, size_dict)


# --- the families ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("dim, n", [(1, 7), (2, 16), (2, 1024), (3, 27), (3, 1000), (4, 16)])
def test_a_lattice_of_side_L_has_L_to_the_dim_tensors_and_dim_L_to_the_dim_minus_1_times_L_minus_1_bonds(dim, n):
    side = round(n ** (1 / dim))
    inputs, output, size_dict = gen("lattice", n, 0, dim=dim).network
    assert len(inputs) == n and len(size_dict) == dim * side ** (dim - 1) * (side - 1)


@pytest.mark.parametrize("degree, n", [(3, 16), (4, 16), (4, 64), (4, 5), (5, 16), (6, 64)])
def test_randreg_gives_every_tensor_degree_bonds(degree, n):
    """n tensors on a random degree-regular graph: n * degree bonds' ends, so n * degree is even (degree 4 allows odd n)."""
    inputs, output, _ = gen("randreg", n, seed=1, degree=degree).network
    assert output == () and len(inputs) == n and all(len(term) == degree for term in inputs)


@pytest.mark.parametrize("n", [2, 16, 64])
def test_a_tree_has_n_minus_1_bonds_and_no_cycle(n):
    inst = gen("tree", n, 0)
    assert len(inst.size_dict) == n - 1 and nx.is_tree(inst.core)


@pytest.mark.parametrize("seed", range(5))
def test_smallworld_contains_its_ring(seed):
    """Newman-Watts only adds shortcuts, so the degree-4 ring is always in the graph."""
    G = FAMILIES["smallworld"](64, seed, degree=4, p=0.5)
    assert all(G.has_edge(i, (i + d) % 64) for i in range(64) for d in (1, 2))
    assert len(gen("smallworld", 5, seed=0).inputs) == 5                                          # the smallest n above the degree


def test_geometric_points_lie_in_the_unit_dim_cube_and_a_larger_radius_gives_more_bonds():
    for dim, smaller, larger in [(2, 1.8, 2.0), (3, 1.4, 1.6)]:
        wider, default = gen("geometric", 64, 0, dim=dim, r_scale=larger), gen("geometric", 64, 0, dim=dim, r_scale=smaller)
        assert len(wider.size_dict) > len(default.size_dict) and f"-r_scale{larger}-" in wider.name
    for n in (16, 64, 256):
        G = FAMILIES["geometric"](n, 0, dim=3, r_scale=1.4)
        assert all(len(pos) == 3 and all(0 <= x <= 1 for x in pos) for _, pos in G.nodes(data="pos"))
        inst = gen("geometric", n, 0, dim=3, r_scale=1.4)
        assert inst.components == 1 and inst.params["connected_by"] == "conditioning" and inst.params["attempt"] <= 6
    assert gen("geometric", 64, 0, dim=4, r_scale=1.6).components == 1                             # other dimensions work


@pytest.mark.parametrize("rank", [3, 4, 7])
def test_hyper_adds_round_share_n_hyperedges_of_the_asked_rank_to_its_regular_graph(rank):
    """3n/2 bonds plus round(share n) indices, each on rank distinct tensors; share 0 is randreg."""
    inputs, output, size_dict = gen(HYPER, 64, seed=0, rank=rank).network
    carriers = Counter(e for term in inputs for e in term)
    assert output == () and Counter(carriers.values()) == {2: 96, rank: 6}                        # 96 bonds; round(0.1 * 64) = 6
    assert all(len(set(term)) == len(term) for term in inputs) and len(size_dict) == 102          # no index twice on a tensor
    plain, reg = gen(HYPER, 16, seed=3, hyper_share=0), gen("randreg", 16, seed=3)
    assert plain.inputs == reg.inputs and plain.size_dict == reg.size_dict
    for share, new in [(0, 0), (0.1, 6), (0.5, 32), (1.0, 64)]:
        inst = gen(HYPER, 64, seed=1, hyper_share=share)
        assert len(inst.size_dict) == 96 + new and inst.components == 1


# --- the streams, ids and Instance -----------------------------------------------------------------------

@pytest.mark.parametrize("family", [*FAMILIES, HYPER])
def test_the_same_seed_gives_the_same_instance_and_another_seed_another(family):
    a, b = gen(family, 64, seed=5), gen(family, 64, seed=5)
    assert a.eq == b.eq and a.size_dict == b.size_dict and a.id == b.id
    if family in RANDOM_FAMILIES:
        assert gen(family, 64, seed=0).id != gen(family, 64, seed=1).id


# ids of fixed (family, knobs, n, seed): the generators must keep drawing the same networks
IDS = [
    ("randreg", {"degree": 3}, 16, 0, "9eb84a9af4f65300"), ("randreg", {"degree": 3}, 64, 1, "839813ce6155641c"),
    ("randreg", {"degree": 3}, 1024, 2, "6b93d13cbb292a1a"),
    ("randreg", {"degree": 4}, 16, 0, "28ddeca6c3ece089"), ("randreg", {"degree": 4}, 64, 2, "b481166a3de6b01f"),
    ("randreg", {"degree": 4}, 256, 1, "cd7792a18f57b184"),
    ("geometric", {"dim": 2, "r_scale": 1.8}, 64, 0, "8c22b2714b088dd4"),
    ("geometric", {"dim": 3, "r_scale": 1.4}, 64, 1, "dcbe6f02d383c9af"),
]


@pytest.mark.parametrize("family, knobs, n, seed, expected", IDS)
def test_the_pinned_draws_keep_their_ids(family, knobs, n, seed, expected):
    """The generators keep drawing the same networks."""
    assert gen(family, n, seed, **knobs).id == expected


def test_an_instance_is_rebuilt_bit_for_bit_from_its_params():
    a = gen("geometric", 64, 5, bonds="spread", legs="region", k=6, d_open=3)
    p = a.params
    assert gen("geometric", p["n"], a.seed, p["chi"], p["bonds"], p["legs"], p["k"], p["d_open"], **p["knobs"]) == a
    b = gen("lattice", 64, 1, legs="region", k=4)
    assert gen("lattice", 64, 1, b.params["chi"], b.params["bonds"], b.params["legs"], b.params["k"], b.params["d_open"]) == b


@pytest.mark.parametrize("family, n", [("lattice", 16), ("geometric", 64), ("randreg", 16), (HYPER, 16)])
@pytest.mark.parametrize("bonds", ["uniform", "spread"])
def test_legs_and_bond_rule_change_neither_the_graph_nor_each_other(family, n, bonds):
    """Three independent streams, SeedSequence([seed, stream, attempt]): what one decides, the others do not move."""
    closed = gen(family, n, 2, bonds=bonds)
    for legs, k in [("scattered", 4), ("region", 4), ("mps", 0)]:
        inst = gen(family, n, 2, bonds=bonds, legs=legs, k=k)
        assert [t[:len(c)] for t, c in zip(inst.inputs, closed.inputs)] == closed.inputs        # the graph
        assert {e: inst.size_dict[e] for e in closed.size_dict} == closed.size_dict             # the bonds
        assert inst.params["attempt"] == closed.params["attempt"]
    u, s = gen(family, n, 2, legs="scattered", k=3), gen(family, n, 2, legs="scattered", k=3, bonds="spread")
    assert u.inputs == s.inputs and u.output == s.output and u.size_dict != s.size_dict          # the same graph and legs
    lattice = gen("lattice", 64, 3, bonds="spread")
    assert set(lattice.size_dict.values()) <= {2, 4, 8, 16} and len(set(lattice.size_dict.values())) > 1
    assert lattice.network == gen("lattice", 64, 3, bonds="spread").network
    assert lattice.size_dict != gen("lattice", 64, 4, bonds="spread").size_dict
    assert instances.spread_bonds(lattice.size_dict, instances._bonds_rng(3), log2_range=(5, 5)) == dict.fromkeys(lattice.size_dict, 32)


@pytest.mark.parametrize("family, n, legs, k", [("lattice", 16, "closed", 0), ("tree", 20, "closed", 0), ("randreg", 16, "closed", 0),
                                                ("lattice", 16, "scattered", 5), ("lattice", 16, "region", 5),
                                                ("lattice", 16, "mps", 0), (HYPER, 16, "scattered", 3), (HYPER, 16, "mps", 0)])
def test_each_legs_pattern_puts_its_legs_where_it_says(family, n, legs, k):
    """closed: none, scattered: with replacement, region: a connected block of k tensors, mps: one per tensor; each leg is a new label on one tensor."""
    closed, inst = gen(family, n, 0), gen(family, n, 0, legs=legs, k=k, d_open=3)
    carriers = [[v for v, t in enumerate(inst.inputs) if e in t] for e in inst.output]
    assert len(inst.output) == (n if legs == "mps" else k) == inst.params["k"]
    assert not set(inst.output) & set(closed.size_dict) and set(inst.output) <= set(inst.size_dict)   # new labels
    assert all(len(vs) == 1 for vs in carriers) and all(inst.size_dict[e] == 3 for e in inst.output)  # one tensor, d_open
    assert inst.components == 1 and [t[:len(c)] for t, c in zip(inst.inputs, closed.inputs)] == closed.inputs
    if legs == "closed":
        assert inst.output == () and closed == gen(family, n, 0, legs="closed")
        if family != HYPER:
            assert set(Counter(e for t in inst.inputs for e in t).values()) == {2}                    # every label a bond
    if legs == "mps":
        assert sorted(vs[0] for vs in carriers) == list(range(n))
        assert instances._unused_labels({"a": 2, "c": 2}, 2) == ["d", "e"]                              # a symbol already in use is skipped
    if legs == "region":
        for seed in range(4):
            big = gen("lattice", 64, seed, legs="region", k=9)
            tensors = [next(v for v, t in enumerate(big.inputs) if e in t) for e in big.output]
            assert len(set(tensors)) == 9 and nx.is_connected(simplify(big.graph).subgraph(tensors))
        assert len(gen("tree", 10, 0, legs="region", k=10).output) == 10                          # the whole network
    if legs == "scattered":
        many = gen("tree", 10, 0, legs="scattered", k=200)                                        # 200 legs on 10 tensors
        assert len(many.output) == 200 and len(many.inputs) == 10


def test_log2_output_and_log2_total_are_sums_of_log2_d(recwarn):
    """log2 of the output size and of the sum over all indices: bits."""
    assert gen("lattice", 64, 0).params["log2_output"] == 0.0 and "legs_dominate" not in gen("lattice", 64, 0).params
    assert gen("lattice", 64, 0, legs="scattered", k=3, d_open=4).params["log2_output"] == 3 * 2.0
    assert gen(HYPER, 64, 0, legs="scattered", k=3, chi=2).params["log2_output"] == 3.0
    assert not recwarn.list                                                                       # generate warns about nothing
    assert gen("lattice", 16, 0).params["log2_total"] == 24.0                                      # 4x4: 24 bonds of 1 bit
    assert gen("lattice", 16, 0, legs="scattered", k=5, d_open=4).params["log2_total"] == 24.0 + 5 * 2.0
    spread = gen("lattice", 16, 0, bonds="spread")
    assert spread.params["log2_total"] == sum(np.log2(d) for d in spread.size_dict.values())


def test_log2_total_bounds_the_width_of_every_path():
    """Every intermediate has at most 2**log2_total entries, so no path search is needed to know a cost cannot overflow."""
    inst = gen("randreg", 64, 0, bonds="spread", legs="scattered", k=10)
    tree = ctg.array_contract_tree(*inst.network, optimize="greedy")
    assert tree.contraction_width() <= inst.params["log2_total"]
    assert np.log2(tree.contraction_cost()) <= inst.params["log2_total"] + np.log2(63)
    big = gen("lattice", 1024, 0, legs="mps")                                                     # 2**1024 entries: allowed, no limit
    assert big.params["log2_output"] == 1024.0 and big.params["log2_total"] == 1024.0 + 1984.0


# --- the knobs, the names, the types -------------------------------------------------------------------

SIZE_OF = {"lattice": 16, "randreg": 16, "geometric": 64, "smallworld": 16, "tree": 16, HYPER: 16}


def test_the_knobs_are_stored_in_knobs_order_and_printed_by_name_and_member():
    """Every knob is stored in instances.KNOBS order, whatever order the caller gives them in, and is in the name."""
    for family, n in SIZE_OF.items():
        inst = gen(family, n, 1)
        assert tuple(inst.params["knobs"]) == instances.KNOBS[family]
        assert all(f"-{k}{v}" in inst.name for k, v in inst.params["knobs"].items())
        assert gen(family, n, 1, **dict(reversed(inst.params["knobs"].items()))) == inst
    assert gen("smallworld", 16, 0, degree=6).name == "smallworld-n16-degree6-p0.1-chi2-uniform-s0"
    assert gen(HYPER, 16, 0, rank=4).params["knobs"] == {"degree": 3, "hyper_share": 0.1, "rank": 4}
    assert gen("lattice", 64, 1, legs="region", k=4).name == "lattice-n64-dim2-chi2-uniform-region4-d2-s1"
    lat, geo = gen("lattice", 27, 0, dim=3), gen("geometric", 64, 0, dim=3, r_scale=1.4)
    assert lat.name == "lattice-n27-dim3-chi2-uniform-s0" and geo.name == "geometric-n64-dim3-r_scale1.4-chi2-uniform-s0"
    assert (lat.member, geo.member) == ("lattice-dim3", "geometric-dim3-r_scale1.4")
    assert gen("randreg", 16, 0, degree=3).member == "randreg-degree3" and gen("tree", 16, 0).member == "tree"
    assert Instance(lat.inputs, lat.output, lat.size_dict, "benchmark", {"n": 27, "name": "x"}, None).name == "benchmark-x"
    assert Instance(lat.inputs, lat.output, lat.size_dict, "benchmark", {"n": 27}, None).member == "benchmark"


def test_numpy_integers_and_floats_are_accepted_where_ints_and_reals_are_meant():
    a = gen("tree", np.int64(16), np.int64(3), np.int64(2), legs="scattered", k=np.int64(5), d_open=np.int64(2))
    assert a.id == gen("tree", 16, 3, 2, legs="scattered", k=5, d_open=2).id
    assert gen("smallworld", 16, 0, degree=6, p=np.float64(0.2)).params["knobs"] == {"degree": 6, "p": 0.2}
    for family, n in [("tree", 2), ("hyper", 4), ("randreg", 4), ("lattice", 1), ("geometric", 1)]:       # the smallest sizes
        assert len(gen(family, n, 0).inputs) == n


def test_components_are_counted_on_the_core():
    """A tensor whose only index is a trace is cut off from the rest: its loop joins nothing."""
    inst = Instance([("a", "a"), ("b", "c"), ("b", "c")], (), {"a": 2, "b": 2, "c": 2}, "test", {"n": 3}, 0)
    assert inst.components == 2 and nx.number_connected_components(inst.core) == 2
    ok = lambda s: ([("a", "b", "t", "t"), ("a", "b")], (), {"a": 2, "b": 2, "t": 2})
    assert draw_connected(ok, seed=0)[3] == 0                                                     # connected through two parallel bonds
    cut = lambda s: ([("t", "t"), ("a",), ("a",)], (), {"a": 2, "t": 2})
    with pytest.raises(ValueError):
        draw_connected(cut, seed=0, tries=2)                                                      # the loop-only tensor is isolated


# --- the checks ---------------------------------------------------------------------------------

BAD_ARGS = [
    # id,                    generate() arguments,                            what the message says
    ("family-unknown",       dict(family="bogus", n=16, seed=0),               "unknown family"),
    ("family-not-a-string",  dict(family=["tree"], n=16, seed=0),              "unknown family"),
    ("n-float",              dict(family="tree", n=16.5, seed=0),              "n must be an int"),
    ("n-bool",               dict(family="tree", n=True, seed=0),              "n must be an int"),
    ("n-string",             dict(family="tree", n="16", seed=0),              "n must be an int"),
    ("n-zero",               dict(family="tree", n=0, seed=0),                 "n must be an int >= 2"),
    ("n-negative-lattice",   dict(family="lattice", n=-4, seed=0),             "n must be an int >= 1"),
    ("n-zero-geometric",     dict(family="geometric", n=0, seed=0),            "n must be an int >= 1"),
    ("n-one-tree",           dict(family="tree", n=1, seed=0),                 "n must be an int >= 2"),
    ("n-below-degree-hyper", dict(family="hyper", n=3, seed=0),                "needs n > degree"),
    ("n-not-a-square",       dict(family="lattice", n=15, seed=0),             "not of that form"),
    ("n-not-a-cube",         dict(family="lattice", n=65, seed=0, dim=3),      "not of that form"),
    ("n-odd-randreg",        dict(family="randreg", n=15, seed=0),             r"n \* degree even"),
    ("n-below-degree",       dict(family="smallworld", n=4, seed=0),           "needs n > degree"),
    ("n-below-degree-randreg", dict(family="randreg", n=3, seed=0, degree=3),  "n > degree"),
    ("seed-negative",        dict(family="tree", n=16, seed=-1),               "seed must be an int >= 0"),
    ("seed-string",          dict(family="tree", n=16, seed="a"),              "seed must be an int >= 0"),
    ("seed-float",           dict(family="tree", n=16, seed=1.0),              "seed must be an int >= 0"),
    ("seed-bool",            dict(family="tree", n=16, seed=True),             "seed must be an int >= 0"),
    ("chi-one",              dict(family="tree", n=16, seed=0, chi=1),         "chi must be an int >= 2"),
    ("chi-float",            dict(family="tree", n=16, seed=0, chi=2.5),       "chi must be an int >= 2"),
    ("chi-string",           dict(family="tree", n=16, seed=0, chi="2"),       "chi must be an int >= 2"),
    ("bonds-unknown",        dict(family="tree", n=16, seed=0, bonds="wide"),  "bonds must be one of"),
    ("bonds-not-a-string",   dict(family="tree", n=16, seed=0, bonds=["uniform"]), "bonds must be one of"),
    ("d_open-one",           dict(family="tree", n=16, seed=0, legs="scattered", k=2, d_open=1), "d_open must be an int >= 2"),
    ("d_open-float",         dict(family="tree", n=16, seed=0, legs="scattered", k=2, d_open=2.5), "d_open must be an int >= 2"),
    ("d_open-bool",          dict(family="tree", n=16, seed=0, legs="scattered", k=2, d_open=True), "d_open must be an int >= 2"),
    ("k-float",              dict(family="tree", n=16, seed=0, legs="scattered", k=2.0), "k must be an int"),
    ("k-string",             dict(family="tree", n=16, seed=0, legs="scattered", k="2"), "k must be an int"),
    ("k-bool",               dict(family="tree", n=16, seed=0, legs="scattered", k=True), "k must be an int"),
    ("legs-unknown",         dict(family="tree", n=16, seed=0, legs="half"),   "legs must be one of"),
    ("legs-not-a-string",    dict(family="tree", n=16, seed=0, legs=["mps"]),  "legs must be one of"),
    ("legs-scattered-k-zero", dict(family="tree", n=10, seed=0, legs="scattered", k=0), "needs k >= 1"),
    ("legs-closed-with-k",   dict(family="tree", n=10, seed=0, legs="closed", k=2), "has no legs"),
    ("legs-mps-k-not-n",     dict(family="tree", n=10, seed=0, legs="mps", k=3), "one leg on each"),
    ("legs-region-above-n",  dict(family="tree", n=10, seed=0, legs="region", k=11), r"k <= n"),
    ("knob-on-a-lattice",    dict(family="lattice", n=16, seed=0, foo=1),      "unknown knob"),
    ("knob-misspelt",        dict(family="smallworld", n=16, seed=0, degre=4), r"its knobs: \['degree', 'p'\]"),
    ("knob-hyper-extra",     dict(family="hyper", n=16, seed=0, n_out=2),      "unknown knob"),
    ("knob-hyper-reg",       dict(family="hyper", n=16, seed=0, reg=4),        "unknown knob"),
    ("knob-k_ratio",         dict(family="lattice", n=16, seed=0, legs="scattered", k_ratio=0.5), "unknown knob"),
    ("degree-two",           dict(family="randreg", n=16, seed=0, degree=2),   "degree must be an int >= 3"),
    ("degree-float",         dict(family="randreg", n=16, seed=0, degree=3.0), "degree must be an int >= 3"),
    ("degree-bool",          dict(family="randreg", n=16, seed=0, degree=True), "degree must be an int >= 3"),
    ("degree-string",        dict(family="randreg", n=16, seed=0, degree="3"), "degree must be an int >= 3"),
    ("degree-not-below-n",   dict(family="randreg", n=6, seed=0, degree=6),    "n > degree"),
    ("degree-hyper-two",     dict(family="hyper", n=16, seed=0, degree=2),     "degree must be an int >= 3"),
    ("rank-two",             dict(family="hyper", n=16, seed=0, rank=2),       "rank must be an int in 3..n"),
    ("rank-above-n",         dict(family="hyper", n=16, seed=0, rank=17),      "rank must be an int in 3..n"),
    ("rank-float",           dict(family="hyper", n=16, seed=0, rank=3.0),     "rank must be an int in 3..n"),
    ("hyper_share-negative", dict(family="hyper", n=16, seed=0, hyper_share=-0.1), "hyper_share must be"),
    ("hyper_share-nan",      dict(family="hyper", n=16, seed=0, hyper_share=float("nan")), "hyper_share must be"),
    ("hyper_share-string",   dict(family="hyper", n=16, seed=0, hyper_share="a"), "hyper_share must be"),
    ("r_scale-zero",         dict(family="geometric", n=16, seed=0, r_scale=0), "r_scale must be"),
    ("r_scale-negative",     dict(family="geometric", n=16, seed=0, dim=3, r_scale=-1.0), "r_scale must be"),
    ("r_scale-inf",          dict(family="geometric", n=16, seed=0, r_scale=float("inf")), "r_scale must be"),
    ("r_scale-string",       dict(family="geometric", n=16, seed=0, r_scale="big"), "r_scale must be"),
    ("dim-zero-lattice",     dict(family="lattice", n=16, seed=0, dim=0),      "dim must be an int >= 1"),
    ("dim-float-lattice",    dict(family="lattice", n=16, seed=0, dim=2.0),    "dim must be an int >= 1"),
    ("dim-bool-geometric",   dict(family="geometric", n=16, seed=0, dim=True), "dim must be an int >= 1"),
    ("dim-list-geometric",   dict(family="geometric", n=16, seed=0, dim=[2]),  "dim must be an int >= 1"),
    ("dim-on-a-tree",        dict(family="tree", n=16, seed=0, dim=2),         "unknown knob"),
    ("p-above-one",          dict(family="smallworld", n=16, seed=0, p=1.5),   "p must be a real number"),
    ("p-string",             dict(family="smallworld", n=16, seed=0, p="a"),   "p must be a real number"),
    ("smallworld-degree-0",  dict(family="smallworld", n=16, seed=0, degree=0), "degree must be an int >= 1"),
]
# a knob left out is the caller's mistake: generate() is called as it is, where gen() would fill it in
MISSING = [(f"missing-{family}-{name}", dict(family=family, n=SIZE_OF[family], seed=0, **{k: v for k, v in gen(family, SIZE_OF[family], 0).params["knobs"].items() if k != name}),
            f"needs the knob.*{name}") for family in SIZE_OF for name in instances.KNOBS[family]]


@pytest.mark.parametrize("filled, args, message", [(True, a, m) for _, a, m in BAD_ARGS] + [(False, a, m) for _, a, m in MISSING],
                         ids=[i for i, _, _ in BAD_ARGS] + [i for i, _, _ in MISSING])
def test_a_bad_argument_is_a_value_error_with_its_reason_before_any_draw(filled, args, message, monkeypatch):
    def no_draw(*a, **kw):
        raise AssertionError("the network was drawn")
    monkeypatch.setattr(instances, "draw_connected", no_draw)
    with pytest.raises(ValueError, match=message):
        (gen if filled else instances.generate)(**args)


def test_draw_connected_gives_up_after_its_tries():
    two_scalars = lambda s: ([("a",), ("b",)], (), {"a": 2, "b": 2})                              # never connected
    with pytest.raises(ValueError, match="no connected draw"):
        draw_connected(two_scalars, seed=0, tries=3)


def test_every_family_has_its_knobs_and_its_check():
    """A family added to FAMILIES cannot be forgotten in KNOBS or CHECKS."""
    assert set(instances.KNOBS) == set(FAMILIES) | {HYPER} == set(instances.CHECKS) == set(SIZE_OF)
    assert set(instances.CONNECTED_BY) == set(instances.KNOBS)


# --- the whole suite -------------------------------------------------------------------------------

@pytest.mark.slow
def test_graph_to_tn_inverts_tn_to_graph_on_the_whole_reference_suite(reference):
    for inst in reference:
        G = tn_to_graph(*inst.network)
        back = graph_to_tn(G)
        assert signature(*back) == signature(*inst.network), inst.name
        assert tn_to_graph(*back).number_of_edges() == G.number_of_edges()


def test_graph_to_tn_inverts_tn_to_graph_up_to_label_names(small_suite):
    """Parallel bonds, a trace, a hyperedge, a local index, a scalar, open legs: every kind survives; so does every small instance."""
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4))
    H = tn_to_graph(*graph_to_tn(G, chi=4))
    assert isinstance(H, nx.MultiGraph) and {frozenset(e) for e in H.edges()} == {frozenset(e) for e in G.edges}
    assert H.number_of_edges() == G.number_of_edges() and all(w == 2.0 for *_, w in H.edges(data="weight"))   # log2 4
    N = ([("a", "b", "t", "t", "o"), ("a", "b", "h", "l", "p"), ("h", "x"), ("h", "x"), ()], ("p", "o"),
         {"a": 2, "b": 3, "t": 5, "h": 2, "l": 7, "x": 4, "o": 3, "p": 2})
    assert signature(*graph_to_tn(tn_to_graph(*N))) == signature(*N)
    for inst in small_suite:
        G = tn_to_graph(*inst.network)
        back = graph_to_tn(G)
        assert signature(*back) == signature(*inst.network), inst.name
        assert tn_to_graph(*back).number_of_edges() == G.number_of_edges()


def test_every_family_is_connected_and_says_how(small_suite):
    """construction: attempt 0; conditioning: connected, redrawn (inst.params["connected_by"], attempt)."""
    assert set(instances.CONNECTED_BY.values()) == {"construction", "conditioning"}
    for inst in small_suite:
        assert inst.components == 1 and inst.params["attempt"] < 10, inst.name                    # more would mean a knob in the wrong regime
        assert inst.params["connected_by"] == instances.CONNECTED_BY[inst.family], inst.name
        if inst.params["connected_by"] == "construction":
            assert inst.params["attempt"] == 0, inst.name

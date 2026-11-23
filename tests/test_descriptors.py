"""descriptors.py: the component and split, the primitives, the rows by group, the table, measure and the descriptors file,
and the documentation that cites the module. Answers known by hand first (the table REFERENCE), then the invariants, then
the errors; the slow tests are marked."""
import itertools
import json
import math
import pathlib
import re

import networkx as nx
import numpy as np
import pytest
from conftest import component, gen, graph_component, signature

from tnsched import config, instances
from tnsched import descriptors as D
from tnsched.instances import graph_to_tn


def grid(L):
    """The L x L grid of bonds, vertices 0..L*L-1."""
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(L, L), ordering="sorted")


def bonds(G):
    """The tensors of the closed network of the graph G of bonds, every bond of dimension 2."""
    return graph_to_tn(nx.convert_node_labels_to_integers(G, ordering="sorted"), 2)[0]


def as_component(x):
    """A graph of bonds is its component; a component stays."""
    return x if isinstance(x, D.ConnectedComponent) else graph_component(x)


def row(x, name):
    """The row `name` of a component or of a graph of bonds: a group runs its function and gives the value at the row's place."""
    probe = D.ROWS[name]
    values = probe.function(as_component(x))
    return values[probe.rows.index(name)] if len(probe.rows) > 1 else values


def largest(inst, seed=0):
    """The largest connected component of the core of an Instance."""
    return D.split(inst.core, seed)[0]


def without_seconds(measured):
    return {k: v for k, v in measured.items() if k != "seconds"}


# --- the networks of REFERENCE: every value computed by hand ----------------------------------------------

NETS = {                                                # name -> (the tensors, the dimensions: 2 unless said)
    "path8": (bonds(nx.path_graph(8)), None),
    "ring8": (bonds(nx.cycle_graph(8)), None),
    "grid4": (bonds(grid(4)), None),
    "K6": (bonds(nx.complete_graph(6)), None),
    "K35": (bonds(nx.complete_bipartite_graph(3, 5)), None),
    "index4": ([("h",)] * 4, {"h": 4}),                                          # one index, d = 4, on 4 tensors
    "index4+bond": ([("h", "x"), ("h", "x"), ("h",), ("h",)], {"h": 4, "x": 2}),
    "index5": ([("h",)] * 5, None),
    "ring4 doubled": ([("a", "d", "e"), ("a", "b"), ("b", "c"), ("c", "d", "e")], None),
    "mixed ranks": ([("h", "b"), ("h", "k"), ("h", "k"), ("h", "b")], {"h": 2, "b": 8, "k": 4}),
}
INF = math.inf
# network -> {row: value | (lo, hi) | None | bool}; all d = 2 unless the network says
REFERENCE = {
    "path8": {"n_indices": 7, "diameter": 7, "girth": None, "_degeneracy": 1, "_mmd": 1,
              "_spectral_tw": 0, "tw_lower": 1, "tw_upper": 1, "tw_w_lower": 2.0, "tw_w_upper": 2.0,
              "lambda_2": 2 * (1 - math.cos(math.pi / 8)), "b_upper": 1.0, "planar": True,
              "genus_lower": 0, "forman_min": 0.0},
    "ring8": {"diameter": 4, "girth": 8, "_degeneracy": 2, "_mmd": 2, "_spectral_tw": 1,
              "tw_lower": 2, "tw_upper": 2, "tw_w_lower": 3.0, "tw_w_upper": 3.0, "b_upper": 2.0, "forman_min": 0.0,
              "forman_mean": 0.0},
    "grid4": {"n_indices": 24, "diameter": 6, "girth": 4, "_degeneracy": 2, "_mmd": (2, 5),
              "_spectral_tw": 2, "tw_lower": (2, 5), "tw_upper": (4, 8), "b_upper": 4.0,
              "b_lower": (0, 4), "planar": True, "forman_min": -4.0},
    "K6": {"diameter": 1, "girth": 3, "_degeneracy": 5, "_mmd": 5, "_spectral_tw": 3,
           "tw_lower": 5, "tw_upper": 5, "lambda_2": 6.0, "b_lower": 8.0, "b_upper": 9.0,
           "planar": False, "genus_lower": 1, "forman_mean": 6.0, "forman_min": 6.0},
    "K35": {"_spectral_tw": 2},
    "index4": {"n_indices": 1, "n_incidences": 4, "n_hyperedges": 1, "hyperedge_fraction": 1.0, "diameter": 1, "girth": None,
               "_degeneracy": 1, "_mmd": 1, "_spectral_tw": 0, "tw_lower": 1, "tw_upper": 1,
               "tw_w_lower": 2.0, "tw_w_upper": 2.0, "b_upper": 2.0, "lambda_2_cut": 2.0,
               "b_lower": 16 / 9, "planar": True, "hyperedge_size_max": 4, "ball_dimension": None},
    "index4+bond": {"girth": 2, "diameter": 1},
    "index5": {"planar": True, "diameter": 1, "girth": None, "genus_lower": 0},                # the primal graph would be K_5
    "ring4 doubled": {"diameter": 2, "girth": 4, "n_indices": 4},
    "mixed ranks": {"n_tensors": 4, "n_indices": 3, "n_hyperedges": 1, "log2_d_mean": 2.0},   # (1 + 3 + 2) / 3 over the indices, not over the edges of H
}
PRIVATE = {"_degeneracy": lambda c: D._degeneracy(c.H), "_mmd": lambda c: D._mmd(c.H), "_spectral_tw": D._spectral_tw}   # the methods behind tw_lower


def check(value, expected, where):
    if expected is None or isinstance(expected, bool):
        assert value is expected, where
    elif isinstance(expected, tuple):
        assert expected[0] - 1e-9 <= value <= expected[1] + 1e-9, (where, value)
    else:
        assert value == pytest.approx(expected), (where, value)


@pytest.mark.parametrize("name", list(REFERENCE))
def test_the_reference_table(name):
    """Every value computed by hand on ten small networks."""
    c = component(*NETS[name][:1], sizes=NETS[name][1])
    for descriptor, expected in REFERENCE[name].items():
        check(PRIVATE[descriptor](c) if descriptor in PRIVATE else row(c, descriptor), expected, (name, descriptor))


@pytest.mark.parametrize("n", [6, 9, 12])
def test_the_closed_forms_of_cycles_complete_graphs_and_grids(n):
    """Cycle: girth n, diameter n // 2, lambda_2 = 2 (1 - cos(2 pi / n)); K_m: tw m - 1, lambda_2 m; L x L grid: planar,
    girth 4, diameter 2 (L - 1), lambda_2 = 2 (1 - cos(pi / L)), 2L(L-1) bonds."""
    assert row(nx.cycle_graph(n), "girth") == n and row(nx.cycle_graph(n), "diameter") == n // 2
    assert row(nx.cycle_graph(n), "lambda_2") == pytest.approx(2 * (1 - math.cos(2 * math.pi / n)))
    K = nx.complete_graph(n)
    assert row(K, "tw_lower") == n - 1 == row(K, "tw_upper")
    assert row(K, "lambda_2") == pytest.approx(n) and row(K, "planar") is False and row(K, "genus_lower") > 0
    L = n // 3 + 2
    G = grid(L)
    assert row(G, "planar") is True and row(G, "girth") == 4 and row(G, "genus_lower") == 0
    assert row(G, "diameter") == 2 * (L - 1) == row(G, "diameter_lower")
    assert row(G, "lambda_2") == pytest.approx(2 * (1 - math.cos(math.pi / L)))
    assert L <= row(G, "tw_upper") <= 2 * L and abs(row(G, "b_upper") - L) <= 2
    assert row(G, "n_tensors") == L * L and row(G, "n_indices") == 2 * L * (L - 1) and row(G, "n_incidences") == 4 * L * (L - 1)
    assert row(G, "n_hyperedges") == 0 and row(G, "hyperedge_size_max") == 2 and row(G, "log2_d_mean") == 1.0 and row(G, "log2_d_std") == 0.0
    assert row(G, "degree_max") == 4 and row(G, "degree_mean") == pytest.approx(4 * L * (L - 1) / (L * L))


# --- the component and split -------------------------------------------------------------------------

def test_split_keeps_an_index_where_two_of_its_tensors_stay_and_gives_pieces_largest_first():
    c = component([("h", "x"), ("h", "x"), ("h",), ("h",)])
    [piece] = D.split(c.H, 0, {0, 1, 2})
    assert piece.n == 3 and D.n_hyperedges(piece) == 1 and len(piece.indices) == 2
    [single] = D.split(c.H, 0, {0, 3})                                                             # a hyperedge with one tensor left joins nothing
    assert len(single.indices) == 1 and single.incidence_graph.degree(single.indices[0]) == 2 and D.n_hyperedges(single) == 1
    H = graph_component(nx.path_graph(7)).H
    assert [P.tensors for P in D.split(H, 0, {5, 0, 1, 2, 6})] == [(0, 1, 2), (5, 6)]              # largest first, ties by the smallest tensor
    assert [P.tensors for P in D.split(H, 0, {1, 3, 5})] == [(1,), (3,), (5,)] and D.split(H, 0, set()) == []
    two = nx.convert_node_labels_to_integers(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(4)))
    assert [P.n for P in D.split(two, 7)] == [4, 3] and all(P.seed == 7 for P in D.split(two, 7))


def test_the_id_of_a_component_of_bonds_only_is_its_instances_and_with_hyperedges_it_is_not():
    """A connected, closed network of bonds only: the component is the instance, so the id is the instance's."""
    for family in ["lattice", "randreg", "tree", "geometric", "smallworld"]:
        inst = gen(family, 16, 0)
        [piece] = D.split(inst.core, 0)
        assert piece.id == inst.id, family
    hyper = gen("hyper", 16, 0)                                                                      # a hyperedge vertex is an index again, labelled by graph_to_tn
    assert D.split(hyper.core, 0)[0].id != hyper.id


def test_two_components_of_one_network_differ_and_two_copies_of_one_core_agree():
    inputs = [(f"a{t}", f"a{(t + 1) % 4}") for t in range(4)] + [(f"b{t}", f"b{(t + 1) % 6}") for t in range(6)]
    pieces = D.split(instances.Instance(inputs, (), dict.fromkeys({e for t in inputs for e in t}, 2), "benchmark", {"n": 10, "name": "x"}, None).core, 0)
    assert len(pieces) == 2 and pieces[0].id != pieces[1].id
    two = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    twice = D.split(nx.convert_node_labels_to_integers(two), 0)
    assert [D.measure(c)["n_tensors"] for c in twice] == [3, 3] and twice[0].id == twice[1].id    # the same core twice, the same id


def test_a_component_with_hyperedges_keeps_its_structure_and_sizes():
    inst = gen("hyper", 16, 0, bonds="spread")
    inputs, output, sizes = graph_to_tn(D.split(inst.core, 0)[0].H)
    assert signature(inputs, output, sizes) == signature(inst.inputs, inst.output, inst.size_dict)
    assert D.n_hyperedges(D.split(inst.core, 0)[0]) == len({e for t in inst.inputs for e in t if sum(e in u for u in inst.inputs) > 2})
    one = component([("h",)] * 4)                                                                   # a hyperedge is one index whatever its rank
    assert D.n_hyperedges(one) == 1 and row(one, "diameter") == 1 and row(one, "girth") is None   # plain BFS on the primal graph: 2


# --- the primitives ----------------------------------------------------------------------------------

def test_tensor_distances_count_indices_and_leave_the_index_vertices_out():
    c = component([("h", "x"), ("h",), ("h", "y"), ("y", "z"), ("z",)])                              # 0, 1, 2 on h; bonds 2-3 (y), 3-4 (z)
    assert D.tensor_distances(c, 0) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
    assert D.tensor_distances(component(bonds(nx.path_graph(5))), 0) == {i: i for i in range(5)}


def test_cut_bits_counts_an_index_once_whatever_its_rank():
    c = component(*NETS["index4"][:1], sizes=NETS["index4"][1])                                      # d = 4: 2 bits
    for A in ({0}, {0, 1}, {0, 1, 2}):
        assert D.cut_bits(c, A) == 2.0
    assert D.cut_bits(c, set()) == 0.0 and D.cut_bits(c, {0, 1, 2, 3}) == 0.0
    assert D.cut_bits(component(bonds(nx.cycle_graph(6))), {0, 1, 2}) == 2.0
    big = component([("h",)] * 40)                                                                   # one index on 40 tensors: a whole clique, no silent cap
    assert big.cut_graph.number_of_edges() == 40 * 39 // 2 and D._size(big, "pairs") == 780 and D.cut_bits(big, {0, 1}) == 1.0
    assert row(big, "lambda_2_cut") > 0
    assert D._size(component([("h",)] * 5), "pairs") == 10 and D._size(graph_component(grid(4)), "pairs") == 24
    assert D._size(graph_component(grid(4)), "tensors + pairs") == 40


def test_on_bonds_the_walk_is_the_lazy_walk():
    G = grid(4)
    P = D.walk(graph_component(G)).toarray()
    A = nx.to_numpy_array(G, nodelist=sorted(G))
    assert P == pytest.approx(0.5 * (np.eye(16) + A / A.sum(axis=1, keepdims=True)))


def test_halve_finds_the_bottleneck_of_a_barbell():
    value, A, B = D.halve(graph_component(nx.barbell_graph(10, 0)))                                  # two K_10 joined by one edge
    assert value == 1 and len(A) == len(B) == 10


def test_the_bisection_profile_of_a_grid_follows_its_perimeter():
    sizes, cuts = D.bisection_profile(graph_component(grid(32)))
    assert sizes[0] == 1024 and cuts[0] <= 34
    assert 0.3 <= row(grid(32), "alpha_bisection") <= 0.8                                           # theory 1/2
    assert row(grid(4), "alpha_bisection") is None


def test_spectral_tw_and_mmd_known_values():
    for n in (6, 9, 12):
        assert n / 2 - 1 <= D._spectral_tw(graph_component(nx.complete_graph(n))) <= n - 1
    assert D._spectral_tw(graph_component(nx.path_graph(30))) == 0
    mmd = lambda G: D._mmd(graph_component(G).H)
    assert mmd(nx.balanced_tree(2, 4)) == 1 and mmd(nx.cycle_graph(11)) == 2
    assert all(mmd(nx.complete_graph(n)) == n - 1 for n in (5, 8)) and all(2 <= mmd(grid(L)) <= 5 for L in (4, 6, 8))
    assert D._mmd(nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4)])) == 2                                    # a graph that is not connected: the minor has isolated vertices


def test_the_T_bracket_of_a_tree_network_is_its_largest_tensor_and_of_a_ring_three_bits():
    """T of a tree network is the largest tensor in bits; of a ring of bonds, 3 bits (two bonds and the cut one)."""
    for rule in ("uniform", "spread"):
        inst = gen("tree", 40, 2, bonds=rule)
        biggest = max(sum(math.log2(inst.size_dict[e]) for e in term) for term in inst.inputs)
        measured = D.measure(largest(inst))
        assert measured["tw_w_lower"] == pytest.approx(biggest) == pytest.approx(measured["tw_w_upper"])
    for n in (3, 5, 9):
        measured = D.measure(component([(f"b{t}", f"b{(t + 1) % n}") for t in range(n)]))
        assert measured["tw_w_lower"] == 3.0 == measured["tw_w_upper"]


@pytest.mark.parametrize("name, G, expected", [
    ("K5", nx.complete_graph(5), 1), ("K33", nx.complete_bipartite_graph(3, 3), 1),
    ("Petersen", nx.petersen_graph(), 1), ("Heawood", nx.heawood_graph(), 1),
    ("grid", grid(8), 0), ("tree", nx.balanced_tree(2, 4), 0), ("6 cubed", nx.grid_graph([6, 6, 6]), 28)])
def test_genus_lower_known_values(name, G, expected):
    """Euler's bound on the genus with the girth of the incidence graph: (m (g - 2) - g (n - 2)) / (2 g), rounded up."""
    assert row(G, "genus_lower") == expected


@pytest.mark.parametrize("G", [nx.petersen_graph(), grid(6), nx.complete_graph(7), nx.wheel_graph(9)])
def test_the_forman_identity(G):
    """mean F = 4 - sum d^2 / m + 9 t / m [For03]."""
    m = G.number_of_edges()
    expected = 4 - sum(d * d for _, d in G.degree()) / m + 9 * sum(nx.triangles(G).values()) / 3 / m
    assert row(G, "forman_mean") == pytest.approx(expected)


# --- invariants of the primitives ---------------------------------------------------------------------

def test_the_cut_graph_never_cuts_more_than_the_cut():
    """A clique of weight w / (floor(r/2) ceil(r/2)) per pair of an index of r tensors cuts at most w."""
    for seed in range(5):
        c = largest(gen("hyper", 12, seed))
        rng = np.random.default_rng(seed)
        for _ in range(60):
            A = {t for t in c.tensors if rng.random() < 0.5}
            assert nx.cut_size(c.cut_graph, A, weight="weight") <= D.cut_bits(c, A) + 1e-9


def test_the_walk_is_stochastic_stays_on_the_tensors_and_is_reversible():
    inputs = [(f"c{i}", f"c{(i + 1) % 16}") + (("h",) if i % 2 == 0 else ()) for i in range(16)]    # a 16-cycle + a hyperedge on 8
    c = component(inputs)
    P = D.walk(c).toarray()
    assert P.shape == (16, 16) and P.min() >= 0 and P.sum(axis=1) == pytest.approx(np.ones(16))
    degree = np.array([c.incidence_graph.degree(t) for t in c.tensors], dtype=float)
    pi = degree / degree.sum()
    assert pi @ P == pytest.approx(pi)                                                              # stationary: pi ~ indices on a tensor


def test_lambda_2_cut_is_lambda_2_on_one_bit_bonds_and_scales_with_the_weights():
    G = grid(5)
    assert row(G, "lambda_2_cut") == pytest.approx(row(G, "lambda_2"))
    nx.set_edge_attributes(G, 3.0, "weight")
    assert row(G, "lambda_2_cut") == pytest.approx(3 * row(G, "lambda_2"))
    assert D._laplacian_spectrum(nx.empty_graph(1), None)[0] == 0.0                                  # one vertex: lambda_2 = 0


@pytest.mark.parametrize("seed", range(20))
def test_b_lower_is_two_ninths_lambda_2_cut_n_and_bounds_every_balanced_cut(seed):
    """b_lower = 2 lambda_2_cut n / 9 <= the smallest cut with both sides of at least n/3: brute force on 12 tensors."""
    assert row(grid(6), "b_lower") == pytest.approx(2 * row(grid(6), "lambda_2_cut") * 36 / 9)
    c = largest(gen("hyper", 12, seed))
    measured = D.measure(c)
    assert c.n == 12 and measured["b_lower"] is not None and "b_lower" not in measured["skipped"]       # defined with hyperedges
    cuts = {A: D.cut_bits(c, A) for size in range(4, 9) for A in itertools.combinations(c.tensors, size)}
    assert row(c, "b_lower") <= min(cuts.values()) + 1e-9
    assert row(c, "b_upper") >= min(v for A, v in cuts.items() if len(A) == 6) - 1e-9


def test_halve_starts_from_the_weighted_fiedler_vector_of_lambda_2_cut():
    c = component([("h", "x"), ("h", "x"), ("h", "y"), ("y", "z"), ("z",)])                          # 0, 1, 2 on h; bonds 0-1, 2-3, 3-4
    D.halve(c)
    assert ("cut", "weight") in c._cache and ("cut", None) not in c._cache


def test_the_brackets_are_ordered_on_the_small_suite(small_suite):
    """tw_lower <= tw_upper, tw_w_lower <= tw_w_upper, b_lower <= b_upper; the largest tensor in bits is below tw_w_lower."""
    seen = set()
    for inst in (i for i in small_suite if i.params["legs"] == "closed"):
        c = largest(inst)
        if c.id in seen:
            continue
        seen.add(c.id)
        measured = D.measure(c)
        methods = (D._degeneracy(c.H), D._mmd(c.H), D._spectral_tw(c))
        biggest = max(sum(math.log2(inst.size_dict[e]) for e in term) for term in inst.inputs)
        assert measured["tw_lower"] == max(methods) <= measured["tw_upper"], inst.name
        assert measured["b_lower"] <= measured["b_upper"] + 1e-9 and measured["b_lower"] <= measured["tw_w_upper"] + 1e-9, inst.name
        assert biggest - 1e-9 <= measured["tw_w_lower"] <= measured["tw_w_upper"] + 1e-9, inst.name


def test_the_double_sweep_never_exceeds_the_diameter():
    for seed in range(3):
        c = largest(gen("geometric", 64, seed))
        assert row(c, "diameter_lower") <= row(c, "diameter")
    c = largest(gen("randreg", 16, 0, degree=3))
    assert row(c, "diameter") == nx.diameter(c.H)                                                   # without hyperedges: the networkx one


# --- the dimension rows --------------------------------------------------------------------------------

BALL_ROWS = ("ball_exponential", "ball_dimension", "ball_growth_rate")
WALK_ROWS = ("walk_exponential", "walk_dimension", "walk_decay_rate")


def law(x, rows):
    """(flag, power value, exponential value) of one group of dimension rows."""
    return D.ROWS[rows[0]].function(as_component(x))


def test_lattices_and_rings_grow_as_power_laws_with_their_dimension():
    for rows in (BALL_ROWS, WALK_ROWS):
        exponential, dimension, rate = law(grid(30), rows)
        assert exponential is False and 1.0 < dimension < 2.6 and rate is None, rows
        exponential, dimension, rate = law(nx.cycle_graph(512), rows)
        assert exponential is False and 0.8 < dimension < 1.2 and rate is None, rows
    assert law(nx.path_graph(2), BALL_ROWS) == (None, None, None)                                    # no source has 4 radii
    assert 1.6 < row(grid(32), "walk_dimension") < 2.4
    assert row(nx.cycle_graph(4096), "walk_dimension") == pytest.approx(1.0, abs=0.05)
    assert row(nx.convert_node_labels_to_integers(nx.grid_2d_graph(64, 64, periodic=True)), "walk_dimension") == pytest.approx(2.0, abs=0.2)    # 2.160 measured
    assert row(nx.convert_node_labels_to_integers(nx.grid_graph([16] * 3, periodic=True)), "walk_dimension") == pytest.approx(3.0, abs=0.3)   # 3.26 measured
    assert row(nx.complete_graph(8), "walk_dimension") is None                                  # too small for a window


def test_expanders_grow_exponentially_and_have_no_dimension():
    for seed in range(3):
        c = largest(gen("randreg", 1024, seed, degree=3))
        for rows in (BALL_ROWS, WALK_ROWS):
            exponential, dimension, rate = law(c, rows)
            assert exponential is True and dimension is None and rate > 0, (seed, rows)
        assert row(c, "walk_dimension") is None


def test_the_ball_regime_is_the_majority_and_a_tie_is_power(monkeypatch):
    c = graph_component(grid(12))
    votes = iter([("exponential", 0.5), ("power", 2.0), ("power", 1.8), ("exponential", 0.7), (None, None)] + [(None, None)] * 3)
    monkeypatch.setattr(D, "_fit_both", lambda x, y, least: next(votes))
    assert law(c, BALL_ROWS) == (False, pytest.approx(1.9), None)                                    # 2 against 2: power, the mean of its two
    c = graph_component(grid(12))
    votes = iter([("exponential", 0.5), ("power", 2.0), ("exponential", 0.7)] + [(None, None)] * 5)
    assert law(c, BALL_ROWS) == (True, None, pytest.approx(0.6))                                     # 2 against 1: exponential


def test_the_fit_tells_the_two_laws_apart_and_refuses_too_few_points():
    x = np.arange(1, 11, dtype=float)
    assert D._fit_both(x, 2 * np.log(x) + 1, 6) == ("power", pytest.approx(2.0))
    assert D._fit_both(x, 0.7 * x + 1, 6) == ("exponential", pytest.approx(0.7))                    # the log of an exponential is a line in x
    assert (D.BALL_MIN_POINTS, D.WALK_MIN_POINTS) == (4, 6)                                          # balls have at most ecc/2 radii
    for least in (D.BALL_MIN_POINTS, D.WALK_MIN_POINTS):
        few = x[:least]
        assert D._fit_both(few, 2 * np.log(few), least) == ("power", pytest.approx(2.0))
        assert D._fit_both(few[:-1], 2 * np.log(few[:-1]), least) == (None, None)


def test_each_dimension_group_gives_one_value_or_none():
    for x in (grid(8), nx.cycle_graph(64), nx.complete_graph(8), nx.path_graph(3)):
        c = as_component(x)
        for rows in (BALL_ROWS, WALK_ROWS):
            exponential, dimension, rate = law(c, rows)
            if exponential is None:
                assert dimension is None and rate is None, rows
            else:
                assert (dimension is None) == exponential and (rate is None) == (not exponential), rows
        assert not c._cache or all(isinstance(key, tuple) for key in c._cache)                       # only gap()'s spectra


def test_the_return_probability_never_rises_and_the_walk_stops_when_mixed():
    """p_k is non-increasing (P is similar to a positive semidefinite matrix), so the walk may stop at the first step
    below MIX_FACTOR pi: every step before it is in the window, none after it would be."""
    for c in (graph_component(grid(8)), graph_component(nx.cycle_graph(64)), component(bonds(nx.complete_graph(8))),
              largest(gen("randreg", 64, 0)), largest(gen("hyper", 64, 0))):
        p, pi = D._return_probability_mean(c)
        assert np.all(np.diff(p) <= 1e-12) and p.min() > pi - 1e-12
        assert np.all(p[:-1] >= D.MIX_FACTOR * pi)
        assert len(p) == D.WALK_MAX_STEPS or p[-1] < D.MIX_FACTOR * pi


def test_the_walk_is_longer_on_a_ring_than_on_an_expander():
    ring, _ = D._return_probability_mean(graph_component(nx.cycle_graph(512)))
    expander, _ = D._return_probability_mean(largest(gen("randreg", 1024, 0)))
    assert len(ring) == D.WALK_MAX_STEPS and len(expander) < 100


@pytest.mark.parametrize("seed", range(3))
def test_on_random_trees_the_walk_dimension_reads_between_1_2_and_1_5(seed):
    """Known limit: a random tree has d_s = 4/3 in the limit, and the row reads 1.29 to 1.46 at n = 1024, one tree at a time."""
    assert 1.2 < row(largest(gen("tree", 1024, seed)), "walk_dimension") < 1.5


@pytest.mark.parametrize("p", [0.05, 0.1])
def test_on_a_small_world_the_balls_say_exponential_and_the_walk_power(p):
    """Known limit: the two rows measure other scales (the balls up to the eccentricity / 2, the walk up to the mixing),
    so on a small world of 1024 tensors they disagree, and the walk's d_s is not a dimension."""
    c = largest(gen("smallworld", 1024, 0, degree=4, p=p))
    assert law(c, BALL_ROWS)[0] is True
    exponential, dimension, _ = law(c, WALK_ROWS)
    assert exponential is False and 2.0 < dimension < 2.7


def test_alpha_of_a_2d_lattice_is_within_two_errors_of_one_half():
    c = largest(gen("lattice", 1024, 0, dim=2))
    alpha, err = D.alpha(c)
    assert err > 0 and abs(alpha - 0.5) <= 2 * err, (alpha, err)                                     # measured 0.500 +- 0.068
    assert 0.01 <= alpha <= 1 and row(c, "alpha_bisection_err") == err


def test_alpha_err_is_the_least_squares_covariance_of_the_fit():
    """sigma^2 (J^T J)^-1 at the optimum, sigma^2 = residual / (points - 3): the error scipy's curve_fit gives on the same data."""
    sizes, cuts = [1024, 512, 256, 128, 64], [30, 20, 14, 10, 7]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(D, "bisection_profile", lambda c: (sizes, cuts))
        alpha, err = D.alpha(None)
    assert alpha == pytest.approx(0.6245, abs=1e-4) and err == pytest.approx(0.036908137, rel=1e-6)    # curve_fit: 0.036908137


def test_alpha_err_is_none_on_three_points_at_a_bound_of_the_search_and_on_a_flat_profile(monkeypatch):
    """Known limit: three parameters fit three points exactly, so there is no residual for an error; at a bound of
    ALPHA_BOUNDS the minimum is the edge of the search; a flat profile has scale 0 and alpha changes nothing."""
    monkeypatch.setattr(D, "bisection_profile", lambda c: ([100, 100, 50], [20, 12, 7]))              # fewer than three distinct sizes
    assert D.alpha(None) == (None, None)
    monkeypatch.setattr(D, "bisection_profile", lambda c: ([100, 50, 25], [20, 12, 7]))               # exactly three points
    assert D.alpha(None)[0] is not None and D.alpha(None)[1] is None
    monkeypatch.undo()
    alpha, err = D.alpha(largest(gen("geometric", 256, 0, dim=2, r_scale=1.8)))                      # b(m) does not grow: dense clumps
    assert alpha == pytest.approx(D.ALPHA_BOUNDS[0], abs=D.BOUND_TOL) and err is None
    alpha, err = D.alpha(graph_component(nx.path_graph(64)))                                          # a tree whose cuts are all one bond
    assert alpha is not None and err is None


# --- the table ----------------------------------------------------------------------------------------

def test_every_row_belongs_to_one_probe_and_every_column_takes_an_allowed_value():
    for probe in D.DESCRIPTORS:
        name = probe.function.__name__
        assert callable(probe.function) and probe.kind in D.KINDS and probe.bounds in D.BOUNDS, name
        assert probe.on in D.ON and probe.rows and all(isinstance(r, str) for r in probe.rows), name
        assert probe.source == "-" or all(re.fullmatch(r"[A-Za-z]+[0-9]{2}", key) for key in probe.source.split(", ")), name
        assert probe.max_size is None or probe.max_size > 0, name
        assert isinstance(probe.complexity, str) and probe.complexity.strip(), name
        doc = probe.function.__doc__ or ""
        assert "\n" not in doc and "docs/descriptors.md, " in doc, name                              # one line, pointing at the docs
        cited = re.findall(r"\[([^\]]+)\]\.$", doc)                                                  # the keys of the source, if any
        assert cited == ([] if probe.source == "-" else [probe.source]), name
    rows = [r for probe in D.DESCRIPTORS for r in probe.rows]
    assert len(rows) == len(set(rows)) == len(D.ROWS)                                                # a row belongs to one probe
    text = D.table()
    assert all(f"`{name}`" in text for name in D.ROWS) and len(text.strip().splitlines()) == len(D.DESCRIPTORS) + 2
    header = [cell.strip() for cell in text.splitlines()[0].strip("|").split("|")]
    assert header.index("complexity") + 1 == header.index("cost")
    assert f"| {D.ROWS['diameter'].complexity} | slow, <= {D.ROWS['diameter'].max_size:,} tensors |" in text
    by = lambda b: {r for r, p in D.ROWS.items() if p.bounds == b}
    assert by("b") == {"b_lower", "b_upper"} and by("W") == set() and by("T") == {"tw_w_lower", "tw_w_upper"}
    assert by("tw") == {"tw_lower", "tw_upper"} and D.ROWS["tw_w_lower"].on == D.ROWS["tw_w_upper"].on == "index graph"
    assert by("D") == set(D.ROWS["ball_dimension"].rows + D.ROWS["walk_dimension"].rows)
    assert D.ROWS["ball_dimension"].rows == BALL_ROWS and D.ROWS["walk_dimension"].rows == WALK_ROWS
    assert D.ROWS["n_tensors"].rows == ("n_tensors", "n_indices", "n_incidences", "log2_d_mean", "log2_d_std")


def test_the_module_defines_the_probes_in_the_order_of_the_table():
    lines = [probe.function.__code__.co_firstlineno for probe in D.DESCRIPTORS]
    assert lines == sorted(lines)
    names = [p.rows[0] for p in D.DESCRIPTORS]
    assert names.index("lambda_2") + 1 == names.index("lambda_2_cut")                                # the two gaps together
    assert names[names.index("ball_exponential") + 1] == "walk_exponential"                          # the dimension groups, together


# --- measure --------------------------------------------------------------------------------------------

def test_a_row_is_the_id_the_rows_skipped_and_seconds_and_is_deterministic():
    inst = gen("smallworld", 64, 2)
    c = largest(inst, 3)
    a, b = D.measure(c), D.measure(largest(inst, 3))
    assert without_seconds(a) == without_seconds(b) and a["id"] == c.id == inst.id
    assert set(a) == {"id", "skipped", "seconds", *D.ROWS} and "total" in a["seconds"]
    assert not {"version", "table", "name", "n_components", "legs", "fingerprint"} & set(a)


def test_open_legs_change_no_row_and_no_id():
    for name, (inputs, sizes) in NETS.items():
        base = D.measure(component(inputs, (), sizes))
        legged = [tuple(t) + (f"leg{i}",) for i, t in enumerate(inputs)]
        output = [f"leg{i}" for i in range(len(inputs))]
        with_legs = D.measure(component(legged, output, dict(sizes or dict.fromkeys({e for t in inputs for e in t}, 2), **dict.fromkeys(output, 2))))
        assert without_seconds(with_legs) == without_seconds(base), name
    plain = D.measure(largest(gen("randreg", 64, 1)))
    for legs, k in [("scattered", 4), ("region", 5), ("mps", 0)]:
        assert without_seconds(D.measure(largest(gen("randreg", 64, 1, legs=legs, k=k)))) == without_seconds(plain), legs


@pytest.mark.parametrize("unit", ["tensors", "indices", "pairs", "tensors + pairs", "edges"])
def test_a_probe_above_its_cap_is_none_and_skipped_in_its_own_unit(unit, monkeypatch):
    """max_size counts tensors, indices, edges of H, pairs of tensors of the cut graph, or tensors + pairs: a probe whose
    unit is `unit` and whose cap is one below the component's size in that unit is skipped, and at the size it runs."""
    c = largest(gen("lattice", 64, 0))
    size = D._size(c, unit)
    capped = {r for p in D.DESCRIPTORS if p.unit == unit for r in p.rows}
    assert capped
    for cap in (size - 1, size):
        monkeypatch.setattr(D, "DESCRIPTORS", tuple(p._replace(max_size=cap) if p.unit == unit else p for p in D.DESCRIPTORS))
        measured = D.measure(c)
        assert set(measured["skipped"]) == (capped if cap < size else set())
        assert all(measured[r] is None for r in measured["skipped"])
    assert D._size(c, "tensors") == 64 and D._size(c, "indices") == 112 == D._size(c, "edges")


def test_slow_false_runs_only_the_fast_probes():
    measured = D.measure(largest(gen("lattice", 64, 0)), slow=False)
    slow = {r for r, p in D.ROWS.items() if not p.fast}
    assert set(measured["skipped"]) == slow and all(measured[n] is None for n in slow)
    assert 2 <= measured["tw_lower"] <= 8 and measured["tw_w_lower"] is not None                      # both lower bounds are fast
    assert D.table_id(slow=False) != D.table_id(slow=True)                                           # the fast file is not the full file


def test_a_row_is_json_able_and_finite_even_on_one_tensor():
    text = json.dumps(D.measure(largest(gen("hyper", 64, 0, legs="mps"))))
    assert "NaN" not in text and "Infinity" not in text
    assert set(D.ROWS) <= set(D.measure(largest(gen("hyper", 64, 0))))
    one = D.measure(component([("a", "a")], sizes={"a": 2}))                                          # one tensor, a trace
    assert one["n_tensors"] == 1 and one["lambda_2"] is None and one["girth"] is None


def disconnected():
    return D.ConnectedComponent(nx.convert_node_labels_to_integers(nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))))


def zero_weight():
    G = nx.path_graph(3)
    G[0][1]["weight"] = 0.0
    return D.ConnectedComponent(G)


@pytest.mark.parametrize("bad, message", [
    (disconnected, r"not connected.*build it with split\(\)"), (lambda: D.ConnectedComponent(nx.Graph()), "no tensor"),
    (zero_weight, "weight"), (lambda: D.measure("not a component"), "ConnectedComponent"), (lambda: D.measure(None), "ConnectedComponent"),
    (lambda: D.measure(largest(gen("randreg", 16, 0)), slow=1), "slow"), (lambda: D.measure(largest(gen("randreg", 16, 0)), slow=None), "slow"),
    (lambda: D.measure(largest(gen("randreg", 16, 0)), slow="yes"), "slow")],
    ids=["disconnected", "empty", "weight-zero", "string", "none", "slow-int", "slow-none", "slow-string"])
def test_a_bad_component_or_argument_is_a_value_error_with_its_reason(bad, message):
    with pytest.raises(ValueError, match=message):
        bad()


# --- the descriptors file ---------------------------------------------------------------------------------

def test_save_appends_rows_and_load_reads_them_by_id(tmp_path):
    """save appends one row to the file (the meta is the script's); load gives {component id: row}."""
    path = tmp_path / "d" / "x.jsonl"
    rows = [D.measure(largest(gen("randreg", 16, 0))), D.measure(largest(gen("tree", 16, 0)))]
    config.save_meta(path, table_id=D.table_id(), seed=0)
    D.save(path, rows[0])
    D.save(path, rows[1])
    loaded = D.load(path)
    assert list(loaded) == [r["id"] for r in rows] and loaded[rows[1]["id"]] == rows[1]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2 and len(D.table_id()) == 16


@pytest.mark.parametrize("what, load_args", [("table_id", {"slow": False}), ("seed", {"seed": 7})])
def test_a_file_of_another_table_slow_or_seed_is_refused_and_says_what_to_do(tmp_path, what, load_args):
    path = tmp_path / "x.jsonl"
    config.save_meta(path, table_id=D.table_id(), seed=0)
    D.save(path, D.measure(largest(gen("tree", 16, 0))))
    with pytest.raises(ValueError, match=what + ".*--fresh"):
        D.load(path, **load_args)


def test_a_file_without_its_meta_is_refused_with_what_to_do(tmp_path):
    path = tmp_path / "x.jsonl"
    with pytest.raises(FileNotFoundError, match="02_compute_descriptors"):
        D.load(path)
    D.save(path, D.measure(largest(gen("tree", 16, 0))))
    with pytest.raises(ValueError, match="no meta file.*--fresh"):
        D.load(path)


# --- the documentation ----------------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
CITATION = re.compile(r"\[([A-Za-z]+[0-9]{2}(?:, [A-Za-z]+[0-9]{2})*)\]")                              # [BKW06] or [BKW06, GHN24]


def published_files():
    """The .py and .md files .gitignore lists to publish (not the tables of results/, which the scripts write), and
    docs/*.md: a file listed and missing fails the test."""
    listed = re.findall(r"^!/(.+\.(?:py|md))$", (ROOT / ".gitignore").read_text(encoding="utf-8"), re.M)
    return sorted({ROOT / name for name in listed if not name.startswith("results/")} | set((ROOT / "docs").glob("*.md")))


def test_every_key_cited_in_the_published_files_is_in_the_readme_references_and_back():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    defined = set(re.findall(r"^\[([A-Za-z]+[0-9]{2})\]", readme.split("## References")[1], re.M))
    cited = {key for f in published_files() for group in CITATION.findall(f.read_text(encoding="utf-8")) for key in group.split(", ")}
    assert cited - defined == set(), f"cited and not in the References: {cited - defined}"
    assert defined - cited == set(), f"in the References and never cited: {defined - cited}"


def test_the_table_in_docs_is_the_table_of_the_module():
    text = (ROOT / "docs" / "descriptors.md").read_text(encoding="utf-8")
    assert text.rstrip().endswith(D.table())


def test_every_docs_pointer_of_a_docstring_names_a_section_of_its_page():
    """"docs/<page>.md, <name>" in a docstring: <name> is a heading ("The estimates" is "estimates") or a bold label."""
    pointers = set()
    for f in (ROOT / "src" / "tnsched").glob("*.py"):
        pointers |= set(re.findall(r"docs/([a-z]+\.md), ([A-Za-z]+(?: [A-Za-z]+)*?)(?= \[|\.|\"|$)", f.read_text(encoding="utf-8"), re.M))
    assert pointers
    for page, name in pointers:
        text = (ROOT / "docs" / page).read_text(encoding="utf-8")
        names = {re.sub(r"^the ", "", h.lower()) for h in re.findall(r"^#+ (.+)$", text, re.M)}
        names |= {b.lower() for b in re.findall(r"\*\*([^*]+)\*\*", text)}
        assert name.lower() in names, f"docs/{page} has no heading or bold label {name!r}"

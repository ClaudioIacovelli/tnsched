"""features.py: the input of one component (native columns, the rwpe and balls encodings, edges), the features file,
and the path to the model (split_labels, samples, unseen). Hand-made networks first, then the invariants
(renumbering, the dense reference), then the file, the limits and the path to the model."""
import networkx as nx
import numpy as np
import pytest
from conftest import TEST_TARGETS, component, files_of, gen, graph_component

from tnsched import config, descriptors, features, suite
from tnsched.features import NATIVE_FEATURES, RWPE_SCALE, RWPE_STEPS, ComponentInput, component_input
from tnsched.instances import Instance

TWO_BY_TWO = ([("a", "b"), ("b", "c"), ("a", "d"), ("c", "d")], (), {"a": 2, "b": 2, "c": 2, "d": 2})
F_T, F_I, K = len(NATIVE_FEATURES["tensor"]), len(NATIVE_FEATURES["index"]), len(RWPE_STEPS)
E = sum(features.ENCODINGS.values())          # every encoding column: what the features file holds after the native ones


def ring(n, chi=2):
    """n tensors in a ring: tensor t carries bonds t and t+1 (mod n)."""
    return [(f"b{t}", f"b{(t + 1) % n}") for t in range(n)], (), {f"b{t}": chi for t in range(n)}


def full_transition(c):
    """The (T+I) x (T+I) random-walk matrix D^-1 A of the bipartite graph, dense: the reference."""
    B = c.incidence_matrix.toarray()
    T, I = B.shape
    A = np.zeros((T + I, T + I))
    A[:T, T:], A[T:, :T] = B, B.T
    return A / A.sum(1, keepdims=True)


def p_of(x, n_native):
    """The return probabilities back from the stored columns log(p) / RWPE_SCALE."""
    return np.exp(RWPE_SCALE * x[:, n_native:].astype(float))


# --- known answers -----------------------------------------------------------------------------------

def test_the_2x2_example_by_hand():
    """Every tensor has 2 indices and 2 bits; every index 1 bit and 2 tensors (log2 = 1); the edges are the incidences."""
    c = component(*TWO_BY_TWO)
    g = component_input(c)
    scale = {kind: [f.scale for f in NATIVE_FEATURES[kind]] for kind in NATIVE_FEATURES}
    assert g.x_tensor.shape == (4, F_T + E) and g.x_index.shape == (4, F_I + E)
    assert g.x_tensor[0, :F_T].tolist() == pytest.approx([2 / scale["tensor"][0], 2 / scale["tensor"][1]])
    assert g.x_index[0, :F_I].tolist() == pytest.approx([1 / scale["index"][0], 1 / scale["index"][1]])
    B = c.incidence_matrix.tocoo()
    assert g.edge.tolist() == [B.row.tolist(), B.col.tolist()] and g.edge.shape == (2, 8)
    assert sorted(g.edge[0].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3] and sorted(g.edge[1].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_rwpe_on_a_ring_of_four_tensors_is_1_2_3_8_5_16_9_32():
    """The bipartite graph of the ring is the 8-cycle: p_2, p_4, p_6, p_8 of its walk."""
    g = component_input(component(*ring(4)))
    known = np.array([1 / 2, 3 / 8, 5 / 16, 9 / 32])
    assert np.allclose(p_of(g.x_tensor, F_T)[:, :4], known, atol=1e-5)
    assert np.allclose(p_of(g.x_index, F_I)[:, :4], known, atol=1e-5)


def test_on_a_lattice_the_tensor_rwpe_at_2j_steps_is_the_diagonal_of_P_to_the_j():
    """Two steps of the bipartite walk are one step of the walk on the tensors (a lattice has only bonds)."""
    c = descriptors.split(gen("lattice", 16, 0).core, 0)[0]
    P = descriptors.walk(c).toarray()
    g = component_input(c)
    for column, s in enumerate(RWPE_STEPS):
        assert p_of(g.x_tensor, F_T)[:, column] == pytest.approx(np.diag(np.linalg.matrix_power(P, s // 2)), rel=1e-4)


def test_a_hyperedge_is_one_index_node_and_legs_and_traces_are_not_in_the_input():
    g = component_input(component([("h",)] * 4, (), {"h": 4}))
    assert g.x_index.shape == (1, F_I + E) and g.edge.shape == (2, 4)
    assert g.x_index[0, :F_I].tolist() == pytest.approx([2 / NATIVE_FEATURES["index"][0].scale, 2 / NATIVE_FEATURES["index"][1].scale])   # d = 4: 2 bits; 4 tensors: log2 = 2
    closed = component_input(component([("a",), ("a",)], (), {"a": 2}))
    legged = component_input(component([("a", "x"), ("a",)], ("x",), {"a": 2, "x": 4}))                # an open leg
    traced = component_input(component([("a", "t", "t"), ("a",)], (), {"a": 2, "t": 8}))               # a trace
    for g in (legged, traced):
        assert np.array_equal(g.x_index, closed.x_index) and np.array_equal(g.edge, closed.edge)
        assert g.x_tensor.shape == closed.x_tensor.shape


def test_one_tensor_has_floor_columns_and_no_index():
    c = component([("a", "a")], sizes={"a": 2})                                                         # a trace: dropped, no index left
    assert c.n == 1 and len(c.indices) == 0
    g = component_input(c)
    assert g.x_tensor.shape == (1, F_T + E) and g.x_index.shape == (0, F_I + E)
    assert g.x_tensor[0, F_T:F_T + K].tolist() == pytest.approx([np.log(features.RWPE_FLOOR) / RWPE_SCALE] * K)
    assert g.x_tensor[0, F_T + K:].tolist() == [0.0] * len(features.BALL_RADII)                       # |B_r| = 1


# --- invariants ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("network", ["2x2", "hyper", "lattice"])
def test_the_block_routine_equals_dense_powers_of_the_whole_walk(network):
    """diag((D^-1 A)^(2j)) from the T x T block, against the dense (T+I) x (T+I) matrix."""
    c = component(*TWO_BY_TWO) if network == "2x2" else descriptors.split(gen(network, 16, 0).core, 0)[0]
    T = c.n
    P = full_transition(c)
    p_t, p_i = features._return_probability_nodes(c, RWPE_STEPS)
    for column, s in enumerate(RWPE_STEPS):
        diag = np.diag(np.linalg.matrix_power(P, s))
        assert p_t[:, column] == pytest.approx(diag[:T], abs=1e-10)
        assert p_i[:, column] == pytest.approx(diag[T:], abs=1e-10)


def test_renumbering_the_tensors_permutes_their_rows_and_an_automorphism_changes_nothing():
    inputs, output, sizes = gen("hyper", 16, 1).network
    perm = np.random.default_rng(0).permutation(len(inputs))                                            # new tensor i is old tensor perm[i]
    g = component_input(component(inputs, output, sizes))
    h = component_input(component([inputs[p] for p in perm], output, sizes))
    assert np.allclose(h.x_tensor, g.x_tensor[perm], atol=1e-6)
    ring12 = component_input(component(*ring(12)))                                                      # every rotation is an automorphism
    assert np.abs(ring12.x_tensor - ring12.x_tensor[0]).max() < 1e-6 and np.abs(ring12.x_index - ring12.x_index[0]).max() < 1e-6


def test_the_native_columns_are_of_order_one_on_the_small_suite(small_suite):
    low = np.log(features.RWPE_FLOOR) / RWPE_SCALE
    for inst in small_suite:
        g = component_input(descriptors.split(inst.core, 0)[0])
        assert np.abs(g.x_tensor[:, :F_T]).max() < 2 and np.abs(g.x_index[:, :F_I]).max() < 2, inst.name
        rwpe = g.x_tensor[:, F_T:F_T + K]
        assert rwpe.max() <= 1e-6 and rwpe.min() >= low - 1e-6, inst.name                                # log p of a probability
        assert 0 <= g.x_tensor[:, F_T + K:].min() and g.x_tensor[:, F_T + K:].max() < 2, inst.name           # log2 |B| / 10


def test_rwpe_stores_log_p_over_the_scale_with_a_floor():
    c = component(*TWO_BY_TWO)
    p_t, _ = features.rwpe(c)
    P = full_transition(c)
    for column, s in enumerate(RWPE_STEPS):
        p = np.diag(np.linalg.matrix_power(P, s))[:4]
        assert p_t[:, column] == pytest.approx(np.log(np.maximum(p, features.RWPE_FLOOR)) / RWPE_SCALE, abs=1e-6)
    assert np.isfinite(features.rwpe(component([("a",), ("a",)], (), {"a": 2}))[0]).all()              # a walk that never returns: the floor, not log 0


# --- the file and its limits ---------------------------------------------------------------------------

# --- balls ---------------------------------------------------------------------------------------------

def ball_counts(c):
    """|B_r| of ball_sizes back from its log2 / BALL_SCALE: (T, 4), (I, 4)."""
    t, i = features.ball_sizes(c)
    return np.rint(2 ** (t * features.BALL_SCALE)), np.rint(2 ** (i * features.BALL_SCALE))


def test_in_the_middle_of_a_path_the_ball_of_radius_r_has_2r_plus_1_tensors():
    c = graph_component(nx.path_graph(21))
    t, _ = ball_counts(c)
    assert t[c.tensors.index(10)].tolist() == [3, 5, 7, 9]


def test_a_ring_has_smaller_balls_than_a_path_of_the_same_degrees_once_r_reaches_half_its_length():
    ring, _ = ball_counts(graph_component(nx.cycle_graph(6)))
    path = graph_component(nx.path_graph(21))
    middle = ball_counts(path)[0][path.tensors.index(10)]
    assert ring[0, :2].tolist() == middle[:2].tolist()                       # r < 3: the same unfolding tree
    assert ring[0, 2:].tolist() == [6, 6] and middle[2:].tolist() == [7, 9]  # r >= 3: the ring closes, 6 tensors


def test_an_index_of_rank_3_counts_its_3_tensors_at_radius_1():
    c = component([("h", "a"), ("h",), ("h",), ("a", "b"), ("b",)])        # h on 3 tensors, then a chain a, b
    _, i = ball_counts(c)
    h = c.indices.index(("index", "h"))                                    # a hyperedge is the vertex ("index", h)
    assert i[h, 0] == 3 and i[h, 1] == 4                                     # r = 2: one more tensor across a


def test_on_a_torus_every_tensor_has_the_same_balls():
    t, _ = ball_counts(graph_component(nx.grid_2d_graph(10, 10, periodic=True)))
    assert (t == t[0]).all() and t[0].tolist() == [5, 13, 25, 41]          # 1 + 2r(r + 1): r <= 4 does not wrap


@pytest.mark.parametrize("family, n, seed", [("hyper", 16, 1), ("randreg", 32, 0), ("tree", 10, 0)])
def test_the_balls_are_those_of_a_bfs_from_every_node(family, n, seed):
    """The sparse products and the BFS of ball_fit (tensor_distances) count the same tensors."""
    c = descriptors.split(gen(family, n, seed).core, 0)[0]
    t, i = ball_counts(c)
    for k, s in enumerate(c.tensors):
        d = np.array(list(descriptors.tensor_distances(c, s).values()))              # indices crossed to each tensor
        assert t[k].tolist() == [(d <= r).sum() for r in features.BALL_RADII]
    for k, e in enumerate(c.indices):
        d = nx.single_source_shortest_path_length(c.incidence_graph, e)            # a tensor at 2r - 1 crosses r
        d = np.array([v for u, v in d.items() if u in set(c.tensors)])
        assert i[k].tolist() == [(d <= 2 * r - 1).sum() for r in features.BALL_RADII]


def test_columns_take_the_native_ones_then_each_asked_encoding_in_the_file_order():
    assert features.columns("tensor", ()) == list(range(F_T))
    assert features.columns("tensor", ("rwpe",)) == list(range(F_T + K))
    assert features.columns("index", ("balls",)) == list(range(F_I)) + list(range(F_I + K, F_I + E))
    assert features.columns("tensor", ("balls", "rwpe")) == list(range(F_T + E))   # the file order, whatever is asked


def test_an_input_carries_the_id_of_its_component():
    c = component(*TWO_BY_TWO)
    assert component_input(c).id == c.id and len(c.id) == 16


def test_save_then_load_gives_the_same_arrays_without_pickle(tmp_path, monkeypatch):
    """A file of another FEATURES_ID (another column, another RWPE step) is refused, and says to run 03 with --fresh."""
    inputs = [component_input(component(*ring(5))), component_input(component(*TWO_BY_TWO))]
    path = tmp_path / "x" / "f.npz"
    features.save(path, inputs)
    config.save_meta(path, features_id=features.FEATURES_ID)
    back = features.load(path)
    assert list(back) == [x.id for x in inputs] and [type(x) for x in back.values()] == [ComponentInput] * 2
    for x in inputs:
        assert all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(x[1:], back[x.id][1:]))
    with np.load(path, allow_pickle=False) as raw:                                                      # arrays only
        assert f"{inputs[0].id}/x_tensor" in raw.files
    monkeypatch.setattr(features, "FEATURES_ID", "0" * 16)
    with pytest.raises(ValueError, match="features_id.*--fresh"):
        features.load(path)


def test_rwpe_and_load_name_their_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "RWPE_MAX_TENSORS", 8)
    with pytest.raises(ValueError, match="RWPE_MAX_TENSORS = 8"):
        features.rwpe(component(*ring(10)))
    with pytest.raises(FileNotFoundError, match="03_compute_features"):
        features.load(tmp_path / "missing.npz")


# --- the path to the model: split_labels, samples, batch --------------------------------------------------

def two_rings():
    """A network of two disconnected rings (4 and 6 tensors): two components, largest first."""
    inputs = [(f"a{t}", f"a{(t + 1) % 4}") for t in range(4)] + [(f"b{t}", f"b{(t + 1) % 6}") for t in range(6)]
    return Instance(inputs, (), {label: 2 for term in inputs for label in term}, "benchmark", {"n": 10, "name": "two-rings"}, None)



def test_split_labels_give_one_label_per_record_and_by_family_leave_the_family_out():
    records = files_of([gen(f, 16 if f != "tree" else 10, s) for f in ("lattice", "randreg", "tree") for s in range(6)])[0]
    for how, held in (("random", None), ("family", "tree"), ("size", None)):
        labels = features.split_labels(records, how, held)
        assert len(labels) == len(records) and set(labels) <= {"train", "val", "test"}
    labels = features.split_labels(records, "family", "tree")
    assert [label == "test" for label in labels] == [rec["family"] == "tree" for rec in records] and "train" in labels
    big = suite.record(gen("lattice", 1024, 0))
    assert features.SIZE_SPLIT_MAX_N < 1024
    assert features.split_labels([big, records[0]], "size")[0] == "test" and features.split_labels([big, records[0]], "size")[1] != "test"


def test_held_out_takes_a_whole_family_or_one_member_and_refuses_a_name_no_record_has():
    """"geometric" holds out both dims; one member name holds out that member only."""
    insts = [gen("geometric", 16, s, dim=d, r_scale=r) for d, r in ((2, 1.8), (3, 1.4)) for s in range(3)]
    records = files_of(insts + [gen("tree", 10, 0)])[0]
    by_family = features.split_labels(records, "family", "geometric")
    assert [label == "test" for label in by_family] == [rec["family"] == "geometric" for rec in records]
    one = features.split_labels(records, "family", "geometric-dim3-r_scale1.4")
    assert [label == "test" for label in one] == [rec["params"]["knobs"]["dim"] == 3 if rec["family"] == "geometric"
                                                  else False for rec in records]
    with pytest.raises(ValueError, match="geometric-dim2-r_scale1.8"):                         # the message lists the members
        features.split_labels(records, "family", "geometrc")


def test_instances_that_differ_only_in_their_legs_or_bond_rule_get_one_label():
    """The hash is on the graph (family, knobs, n, seed): the variants of one graph are never on two sides."""
    tree = suite.record(gen("tree", 10, 0))                       # a record of the held-out family: held_out must exist
    for how, held in (("random", None), ("family", "tree"), ("size", None)):
        for seed in range(12):
            variants = [suite.record(gen("randreg", 16, seed, bonds=bonds, legs=legs, k=k))
                        for bonds in ("uniform", "spread") for legs, k in (("closed", 0), ("scattered", 3), ("mps", 0))]
            assert len(set(features.split_labels(variants + [tree], how, held)[:-1])) == 1, (how, seed)


def test_random_split_is_about_80_10_10_and_a_benchmark_network_is_hashed_by_its_name():
    labels = features.split_labels([suite.record(gen("tree", 8, seed)) for seed in range(300)])
    share = {k: labels.count(k) / 300 for k in ("train", "val", "test")}
    assert 0.7 < share["train"] < 0.9 and 0.03 < share["val"] < 0.2 and 0.03 < share["test"] < 0.2, share
    named = [suite.record(Instance([("a", "a")], (), {"a": 2}, "benchmark", {"n": 1, "name": f"net{i}", "paths": {}}, None)) for i in range(40)]
    assert len(set(features.split_labels(named))) > 1                                                  # the same n, other names, other buckets


def test_split_labels_refuse_what_they_do_not_know():
    records = files_of([gen("tree", 8, 0)])[0]
    with pytest.raises(ValueError, match="unknown split"):
        features.split_labels(records, "nope")
    with pytest.raises(ValueError, match="held_out"):
        features.split_labels(records, "family")


def test_every_row_is_given_left_out_or_predicted_and_every_target_has_a_kind():
    assert set(features.TARGETS) == set(descriptors.ROWS) - set(features.GIVEN) - set(features.LEFT_OUT)
    assert {name for name, kind in features.TARGETS.items() if kind == "binary"} == set(features.BINARY)
    assert {name for name, kind in features.TARGETS.items() if kind == "linear"} == set(features.LINEAR)
    assert set(features.TARGETS.values()) == {"binary", "linear", "log"}
    assert features.TARGETS["tw_upper"] == "log" and "n_tensors" not in features.TARGETS


def test_the_slow_targets_are_the_targets_of_slow_probes():
    assert set(features.SLOW_TARGETS) == {"diameter", "girth", "tw_upper", "tw_w_upper", "b_upper", "genus_lower"}
    assert all(not descriptors.ROWS[name].fast for name in features.SLOW_TARGETS)


def test_a_disconnected_network_gives_one_sample_per_component_largest_first_with_y_in_the_order_of_targets():
    records, components, rows, inputs = files_of([two_rings()])
    pairs = features.samples(records, components, rows, inputs, TEST_TARGETS)
    assert len(pairs) == 2 and [x.x_tensor.shape[0] for x, _ in pairs] == [6, 4] and [x.id for x, _ in pairs] == list(rows)
    x, y = pairs[1]
    assert x.x_tensor.shape == (4, F_T + E) and y.shape == (len(TEST_TARGETS),) and y.dtype == np.float32
    assert y[0] == rows[x.id]["tw_lower"] and y[1] in (0.0, 1.0)                                      # a binary target is 0 or 1
    assert np.isnan(y[3])                                                                              # alpha_bisection is a slow row: not measured
    rows[pairs[0][0].id]["forman_mean"] = None
    assert np.isnan(features.samples(records, components, rows, inputs, TEST_TARGETS)[0][1][2])             # a missing descriptor is NaN


def test_samples_skip_a_component_above_max_dense_and_name_the_script_for_a_missing_component_row_or_input(monkeypatch):
    records, components, rows, inputs = files_of([two_rings()])
    with pytest.raises(ValueError, match="02_compute_descriptors"):
        features.samples(records, {}, rows, inputs, TEST_TARGETS)
    with pytest.raises(ValueError, match="02_compute_descriptors"):
        features.samples(records, components, {}, inputs, TEST_TARGETS)
    with pytest.raises(ValueError, match="03_compute_features"):
        features.samples(records, components, rows, {}, TEST_TARGETS)
    monkeypatch.setattr(features, "RWPE_MAX_TENSORS", 5)
    pairs = features.samples(records, components, rows, inputs, TEST_TARGETS)
    assert len(pairs) == 1 and pairs[0][0].x_tensor.shape[0] == 4


def test_the_components_file_gives_the_components_of_each_instance_or_names_the_script(tmp_path):
    records, components, _, _ = files_of([two_rings(), gen("tree", 8, 0)])
    path = tmp_path / "x_components.jsonl"
    path.write_text("".join(config.json_line({"instance": i, "components": [list(c) for c in cs]}) + "\n" for i, cs in components.items()),
                    encoding="utf-8")
    assert descriptors.load_components(path) == components and [n for _, n in components[records[0]["id"]]] == [6, 4]
    with pytest.raises(FileNotFoundError, match="02_compute_descriptors"):
        descriptors.load_components(tmp_path / "missing.jsonl")


def test_unseen_leaves_out_the_pairs_whose_component_is_in_train():
    def pair(component_id):
        x = features.ComponentInput(component_id, np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32),
                                    np.zeros((2, 0), np.int64))
        return x, np.zeros(1, np.float32)
    train_pairs, val_pairs = [pair("a"), pair("b")], [pair("b"), pair("c"), pair("c")]
    assert [x.id for x, _ in features.unseen(val_pairs, train_pairs)] == ["c", "c"]
    assert [x.id for x, _ in features.unseen(val_pairs, [])] == ["b", "c", "c"]

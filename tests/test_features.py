"""features.py: the input of one component (native columns, random-walk positional encoding, edges) and the features file.
Hand-made networks first, then the invariants (renumbering, the dense reference), then the file and the limits."""
import numpy as np
import pytest
from conftest import component, gen

from tnsched import config, descriptors, features
from tnsched.features import FEATURES, LOG_SCALE, RWPE_STEPS, ComponentInput, component_input

TWO_BY_TWO = ([("a", "b"), ("b", "c"), ("a", "d"), ("c", "d")], (), {"a": 2, "b": 2, "c": 2, "d": 2})
F_T, F_I, K = len(FEATURES["tensor"]), len(FEATURES["index"]), len(RWPE_STEPS)


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
    """The return probabilities back from the stored columns log(p) / LOG_SCALE."""
    return np.exp(LOG_SCALE * x[:, n_native:].astype(float))


# --- known answers -----------------------------------------------------------------------------------

def test_the_2x2_example_by_hand():
    """Every tensor has 2 indices and 2 bits; every index 1 bit and 2 tensors (log2 = 1); the edges are the incidences."""
    c = component(*TWO_BY_TWO)
    g = component_input(c)
    scale = {kind: [f.scale for f in FEATURES[kind]] for kind in FEATURES}
    assert g.x_tensor.shape == (4, F_T + K) and g.x_index.shape == (4, F_I + K)
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
    assert g.x_index.shape == (1, F_I + K) and g.edge.shape == (2, 4)
    assert g.x_index[0, :F_I].tolist() == pytest.approx([2 / FEATURES["index"][0].scale, 2 / FEATURES["index"][1].scale])   # d = 4: 2 bits; 4 tensors: log2 = 2
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
    assert g.x_tensor.shape == (1, F_T + K) and g.x_index.shape == (0, F_I + K)
    assert g.x_tensor[0, F_T:].tolist() == pytest.approx([np.log(features.LOG_FLOOR) / LOG_SCALE] * K)


# --- invariants ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("network", ["2x2", "hyper", "lattice"])
def test_the_block_routine_equals_dense_powers_of_the_whole_walk(network):
    """diag((D^-1 A)^(2j)) from the T x T block, against the dense (T+I) x (T+I) matrix."""
    c = component(*TWO_BY_TWO) if network == "2x2" else descriptors.split(gen(network, 16, 0).core, 0)[0]
    T = c.n
    P = full_transition(c)
    p_t, p_i = features._return_probabilities(c, RWPE_STEPS)
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
    low = np.log(features.LOG_FLOOR) / LOG_SCALE
    for inst in small_suite:
        g = component_input(descriptors.split(inst.core, 0)[0])
        assert np.abs(g.x_tensor[:, :F_T]).max() < 2 and np.abs(g.x_index[:, :F_I]).max() < 2, inst.name
        assert g.x_tensor[:, F_T:].max() <= 1e-6 and g.x_tensor[:, F_T:].min() >= low - 1e-6, inst.name   # log p of a probability


def test_rwpe_stores_log_p_over_the_scale_with_a_floor():
    c = component(*TWO_BY_TWO)
    p_t, _ = features.rwpe(c)
    P = full_transition(c)
    for column, s in enumerate(RWPE_STEPS):
        p = np.diag(np.linalg.matrix_power(P, s))[:4]
        assert p_t[:, column] == pytest.approx(np.log(np.maximum(p, features.LOG_FLOOR)) / LOG_SCALE, abs=1e-6)
    assert np.isfinite(features.rwpe(component([("a",), ("a",)], (), {"a": 2}))[0]).all()              # a walk that never returns: the floor, not log 0


# --- the file and its limits ---------------------------------------------------------------------------

def test_an_input_carries_the_id_of_its_component():
    c = component(*TWO_BY_TWO)
    assert component_input(c).id == c.id and len(c.id) == 16


def test_save_then_load_gives_the_same_arrays_without_pickle(tmp_path, monkeypatch):
    """A file of another FEATURES_ID (another column, another RWPE step) is refused, and says to run 03 with --fresh."""
    inputs = [component_input(component(*ring(5))), component_input(component(*TWO_BY_TWO))]
    path = tmp_path / "x" / "f.npz"
    features.save(path, inputs)
    config.write_meta(path, features_id=features.FEATURES_ID)
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
    monkeypatch.setattr(features, "MAX_DENSE", 8)
    with pytest.raises(ValueError, match="MAX_DENSE = 8"):
        features.rwpe(component(*ring(10)))
    with pytest.raises(FileNotFoundError, match="03_compute_features"):
        features.load(tmp_path / "missing.npz")

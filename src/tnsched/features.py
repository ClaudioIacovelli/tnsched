"""The model's input: one connected component of a tensor network as a bipartite graph with node features.

The classifier sees exactly the graph on which the descriptors are measured: one ConnectedComponent
(the core of the network: no open legs, no traces, parallel bonds merged, one connected piece). Any
input reaches it through the same door: an Instance (a graph goes through instances.graph_to_tn
first) -> descriptors.split(inst.core, seed) -> one input per component. Open legs and traces are not
in the input: they enter no descriptor.

The module reads top to bottom:
1. FEATURES: the native node columns;
2. rwpe: the one positional encoding, a function of the component alone;
3. component_input: the component -> node columns and edges, the content of the features file;
4. the features file: save, load.
Plain numpy and scipy (the graph views come from descriptors).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import sparse

from . import config
from .descriptors import ConnectedComponent, walk


# === 1. the native columns ==============================================================

class Feature(NamedTuple):
    """One column of a node: a function of the component with one value per node, and its fixed scale."""
    name: str
    function: Callable[[ConnectedComponent], list]
    scale: float                 # the column is divided by it: a constant of this table, not a statistic of
                                 # the data, so that the numbers are of order 1


FEATURES: dict[str, tuple[Feature, ...]] = {
    "tensor": (Feature("degree", lambda c: c.degrees, 8.0),                  # how many indices it carries
               Feature("log2_size", lambda c: [sum(c.w[k] for k in c.incidence_graph[t]) for t in c.tensors], 32.0)),
    "index": (Feature("w", lambda c: [c.w[k] for k in c.indices], 4.0),      # log2 of its dimension
              Feature("log2_rank", lambda c: [math.log2(c.ranks[k]) for k in c.indices], 8.0)),   # log2 of its tensors
}


class ComponentInput(NamedTuple):
    """T tensor nodes, I index nodes, E incidences of one component; k = len(RWPE_STEPS) encoding columns."""
    id: str                     # the component's id (ConnectedComponent.id): the key in the descriptors file too
    x_tensor: np.ndarray        # (T, F_t + k) float32: FEATURES["tensor"] over their scales, then the rwpe columns
    x_index: np.ndarray         # (I, F_i + k) float32: FEATURES["index"] over their scales, then the rwpe columns
    edge: np.ndarray            # (2, E) int64: row 0 the position of the tensor in c.tensors, row 1 the
                                # position of the index in c.indices (the nonzeros of c.incidence_matrix)


# === 2. the random-walk encoding ========================================================

# The walk on the bipartite graph returns to a node only after an even number of steps. Every even step up to
# 16, then about geometric: p_s ~ s^(-d_s/2) is a power law, evenly spaced on a log axis.
RWPE_STEPS = (2, 4, 6, 8, 10, 12, 14, 16, 24, 32, 48, 64, 96, 128)
LOG_FLOOR = 1e-30                        # p = 0 (the walk never comes back) would give log 0
LOG_SCALE = 8.0                          # the stored value is log(p) / LOG_SCALE: a constant of the table

# P^j and the walk on the indices are dense matrices: T x T (32 MB at T = 2048) and T x I. Above this many
# tensors rwpe raises, and script 03 skips the component.
MAX_DENSE = 2048


def _return_probabilities(c: ConnectedComponent, steps: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities of the walk on the bipartite graph after the (even, ascending) `steps`: (T, len)
    for tensor starts, (I, len) for index starts. After 2j steps a tensor returns with probability diag(P^j),
    P = walk(c) the two-step walk on the tensors; an index with diag(X P^(j-1) Y) = diag(Q^j), Q = D_I^-1 B^T
    D_T^-1 B, with X = D_I^-1 B^T and Y = D_T^-1 B. Its own function: the loop is long, rwpe only takes the log."""
    T, I = c.n, len(c.indices)
    if T < 2:
        return np.zeros((T, len(steps))), np.zeros((I, len(steps)))      # one tensor: there is no walk
    B = c.incidence_matrix
    X = sparse.diags(1.0 / np.array([c.ranks[k] for k in c.indices])) @ B.T       # (I, T)
    Y = (sparse.diags(1.0 / np.array(c.degrees)) @ B).toarray()                   # (T, I)
    P = walk(c).toarray()                                                          # (T, T) = Y X
    wanted = {s // 2: column for column, s in enumerate(steps)}
    p_tensor, p_index = np.zeros((T, len(steps))), np.zeros((I, len(steps)))
    power, tail = np.eye(T), Y                                  # P^(j-1) and P^(j-1) Y
    for j in range(1, max(wanted) + 1):
        power_j = power @ P                                      # P^j
        if j in wanted:
            p_tensor[:, wanted[j]] = np.diag(power_j)
            p_index[:, wanted[j]] = np.asarray(X.multiply(tail.T).sum(axis=1)).ravel()   # diag(X P^(j-1) Y)
        power, tail = power_j, P @ tail
    return p_tensor, p_index


def rwpe(c: ConnectedComponent) -> tuple[np.ndarray, np.ndarray]:
    """log p_s(v) / LOG_SCALE for s in RWPE_STEPS, p_s(v) = [(D^-1 A)^s]_vv, v a node of the bipartite graph;
    arrays (T, 14), (I, 14). A fixed function of the component: the same on nodes an automorphism exchanges.
    Reference: [DLL22] (random-walk positional encoding), even steps only (bipartite)."""
    if c.n > MAX_DENSE:
        raise ValueError(f"rwpe is dense: at most MAX_DENSE = {MAX_DENSE} tensors, this component has {c.n}")
    p_tensor, p_index = _return_probabilities(c, RWPE_STEPS)
    return np.log(np.maximum(p_tensor, LOG_FLOOR)) / LOG_SCALE, np.log(np.maximum(p_index, LOG_FLOOR)) / LOG_SCALE


# === 3. the input =======================================================================

def component_input(c: ConnectedComponent) -> ComponentInput:
    """The graph of one component: a node per tensor and per index, an edge where a tensor carries an index.
    Deterministic: the native columns, then rwpe. This is what the features file holds."""
    def features(kind: str) -> np.ndarray:
        columns = np.array([f.function(c) for f in FEATURES[kind]], dtype=np.float32)    # (F, n)
        return columns.T / np.array([f.scale for f in FEATURES[kind]], dtype=np.float32)   # (n, F)
    coo = c.incidence_matrix.tocoo()          # the same B, stored as its nonzeros only
    # coo.row[k], coo.col[k]: the tensor and the index of the k-th nonzero, i.e. of one edge
    edge = np.array([coo.row, coo.col], dtype=np.int64)                                 # (2, E)
    p_tensor, p_index = rwpe(c)
    return ComponentInput(c.id, np.concatenate([features("tensor"), p_tensor.astype(np.float32)], axis=1),
                          np.concatenate([features("index"), p_index.astype(np.float32)], axis=1), edge)


# === 4. the features file ===============================================================

# The identity of what component_input computes: the meta of a features file records it, and a file made with
# another is refused. It sees names, scales and constants, not the body of a function.
FEATURES_ID = hashlib.sha256(repr(([(kind, [(f.name, f.scale) for f in rows]) for kind, rows in FEATURES.items()],
                                   RWPE_STEPS, LOG_SCALE, LOG_FLOOR, MAX_DENSE)).encode()).hexdigest()[:16]


def save(path: Path, inputs: list[ComponentInput]) -> None:
    """results/features/<name>.npz: three arrays per input, "<id>/x_tensor", "<id>/x_index" and "<id>/edge"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{x.id}/{field}": getattr(x, field) for x in inputs for field in ("x_tensor", "x_index", "edge")})


def load(path: Path) -> dict[str, ComponentInput]:
    """{input.id: input} of save(). FileNotFoundError, naming the script to run, if there is no file; ValueError
    if its meta has another FEATURES_ID."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run scripts/03_compute_features.py")
    config.read_meta(path, "Run scripts/03_compute_features.py with --fresh.", features_id=FEATURES_ID)
    with np.load(path, allow_pickle=False) as data:
        return {key: ComponentInput(key, *(data[f"{key}/{field}"] for field in ("x_tensor", "x_index", "edge")))
                for key in dict.fromkeys(name.split("/")[0] for name in data.files)}

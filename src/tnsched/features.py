"""The model's input: one connected component as a bipartite graph (tensor nodes, index nodes) with node columns.
docs/features.md.

The model sees exactly the graph the descriptors are measured on: one ConnectedComponent of inst.core (no open legs,
no traces, parallel bonds merged). Every input comes the same way: an Instance -> descriptors.split(inst.core, seed)
-> one input per component.

Reading order:
1. NATIVE_FEATURES, the native node columns, and ComponentInput;
2. the encodings a model may ask for: rwpe (where a walk returns) and balls (how many tensors are near);
3. ENCODINGS, columns, component_input: a component -> its node columns and edges, what the features file holds;
4. the features file: save, load;
5. the path to the model, for one instances file and the files named after it:
     records (config.load_lines) -> split_labels -> "train" / "val" / "test" per instance
     -> samples(records of one label, ...) -> [(ComponentInput, y)], one per component
     -> unseen(val or test pairs, train pairs).
Plain numpy and scipy, one component at a time: batching is the model's (docs/features.md).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy import sparse

from . import config, descriptors
from .descriptors import ConnectedComponent, walk
from .instances import member_name


# === 1. the native columns ==============================================================

class Feature(NamedTuple):
    """One column of a node: a function of the component with one value per node, and its fixed scale."""
    name: str
    function: Callable[[ConnectedComponent], list]
    scale: float                 # the column is divided by it, so the numbers are of order 1: a constant of this
                                 # table, not a statistic of the data


NATIVE_FEATURES: dict[str, tuple[Feature, ...]] = {
    "tensor": (Feature("degree", lambda c: c.degrees, 8.0),                  # how many indices it carries
               Feature("log2_size", lambda c: [sum(c.w[k] for k in c.incidence_graph[t]) for t in c.tensors], 32.0)),
    "index": (Feature("w", lambda c: [c.w[k] for k in c.indices], 4.0),      # log2 of its dimension
              Feature("log2_rank", lambda c: [math.log2(c.ranks[k]) for k in c.indices], 8.0)),   # log2 of its tensors
}


class ComponentInput(NamedTuple):
    """One component: T tensor nodes, I index nodes, E incidences (edges); k columns of encodings (all of ENCODINGS)."""
    id: str                     # the component's id (ConnectedComponent.id): the key in the descriptors file too
    x_tensor: np.ndarray        # (T, F_t + k) float32: NATIVE_FEATURES["tensor"] over their scales, then the encodings
    x_index: np.ndarray         # (I, F_i + k) float32: NATIVE_FEATURES["index"] over their scales, then the encodings
    edge: np.ndarray            # (2, E) int64: column k is one incidence, (row of x_tensor, row of x_index): the
                                # nonzeros of c.incidence_matrix


# === 2. the encodings ==================================================================

# --- rwpe: the return probabilities of the walk -------------------------------------------

# The walk on the bipartite graph returns to a node only after an even number of steps. Every even step up to 16,
# then about geometric: p_s ~ s^(-d_s/2) is a power law, so its points are spread evenly on a log axis.
RWPE_STEPS = (2, 4, 6, 8, 10, 12, 14, 16, 24, 32, 48, 64, 96, 128)
RWPE_FLOOR = 1e-30                        # p = 0 (the walk never comes back) would give log 0
RWPE_SCALE = 8.0                          # the stored value is log(p) / RWPE_SCALE: a constant of the table

# P^j is dense, T x T (32 MB at T = 2048), and so is P^(j-1) Y, T x I. Above this many tensors rwpe raises and
# script 03 skips the component.
RWPE_MAX_TENSORS = 2048


def _return_probability_nodes(c: ConnectedComponent, steps: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The probability that the walk is back at its start after each of the (even, ascending) `steps`, per node:
    (T, len(steps)) for the tensors, (I, len(steps)) for the indices."""
    T, I = c.n, len(c.indices)
    if T < 2:
        return np.zeros((T, len(steps))), np.zeros((I, len(steps)))      # one tensor: there is no walk
    # The walk alternates two half-steps, each a matrix of probabilities (rows sum to 1):
    #   Y[t, e] = 1/deg(t)   from tensor t to one of its indices, uniformly
    #   X[e, t] = 1/rank(e)  from index e to one of its tensors, uniformly (the one it came from too)
    # After 2j steps a tensor is back with probability diag(P^j), P = Y X; an index with diag(X P^(j-1) Y)
    B = c.incidence_matrix                                                         # (T, I), B[t, e] = 1 if e on t
    X = sparse.diags(1.0 / np.array([c.ranks[k] for k in c.indices])) @ B.T       # (I, T) = D_I^-1 B^T
    Y = sparse.diags(1.0 / np.array(c.degrees)) @ B                               # (T, I) = D_T^-1 B
    P = walk(c)                                                                    # (T, T) = Y X, P[t, u] = p(t -> u)
    wanted = {s // 2: column for column, s in enumerate(steps)}
    p_tensor, p_index = np.zeros((T, len(steps))), np.zeros((I, len(steps)))
    # only the power is dense: P and Y stay sparse, and P^(j-1) Y is formed only at the wanted steps
    power = np.eye(T)                                            # P^0; at step j of the loop it holds P^(j-1)
    for j in range(1, max(wanted) + 1):
        power_j = power @ P                                      # P^j
        if j in wanted:
            p_tensor[:, wanted[j]] = np.diag(power_j)
            tail = power @ Y                                     # (T, I) = P^(j-1) Y
            p_index[:, wanted[j]] = np.asarray(X.multiply(tail.T).sum(axis=1)).ravel()   # diag(X P^(j-1) Y)
        power = power_j
    return p_tensor, p_index


def rwpe(c: ConnectedComponent) -> tuple[np.ndarray, np.ndarray]:
    """log p_s(v) / RWPE_SCALE for s in RWPE_STEPS, p_s(v) = [(D^-1 A)^s]_vv the return probability of node v:
    (T, 14), (I, 14). Equal on nodes a symmetry of the graph exchanges. Random-walk positional encoding [DLL22], even
    steps only (the graph is bipartite)."""
    if c.n > RWPE_MAX_TENSORS:
        raise ValueError(f"rwpe is dense: at most RWPE_MAX_TENSORS = {RWPE_MAX_TENSORS} tensors, "
                         f"this component has {c.n}")
    p_tensor, p_index = _return_probability_nodes(c, RWPE_STEPS)
    return (np.log(np.maximum(p_tensor, RWPE_FLOOR)) / RWPE_SCALE,
            np.log(np.maximum(p_index, RWPE_FLOOR)) / RWPE_SCALE)


# --- balls: the number of tensors near a node ---------------------------------------------

# |B_r(v)|: the tensors reached from node v crossing at most r indices.
# Why: a message-passing network that sums its messages counts walks, not distinct nodes (the 1-WL limit), so it
# does not see short cycles. A ball counts distinct tensors: a cycle of length 2r or less makes it smaller than on a
# tree. It is an exact count, equal on nodes a symmetry exchanges. (ball_fit counts the same balls from 8 sources to
# fit one dimension per component; here every node keeps its own.)
BALL_RADII = (1, 2, 3, 4)       # crossing at most r indices: r = 4 is 8 steps of the incidence graph
BALL_SCALE = 10.0               # the column is log2 |B| / BALL_SCALE: 1 at 1024 tensors


def ball_sizes(c: ConnectedComponent) -> tuple[np.ndarray, np.ndarray]:
    """log2 |B_r(v)| / BALL_SCALE for r in BALL_RADII, v a tensor or an index of c: arrays (T, 4), (I, 4).
    docs/features.md, ball encoding."""
    # All sources, few radii: one sparse boolean product per radius, every node at once. M = B B^T: the tensors
    # that share an index. R_r = (1 + M)^r ("stay, or cross one index", r times): row t holds the tensors within r
    # indices of t, so |B_r(t)| = the nonzeros of row t of R_r.
    B = c.incidence_matrix.astype(bool)                               # (T, I)
    one = sparse.eye_array(c.n, dtype=bool, format="csr")
    step = (one + B @ B.T).tocsr()                                    # (T, T) = 1 + M
    reach = one                                                       # R_0: a tensor reaches itself
    tensor_sizes, index_sizes = [], []
    for r in range(1, max(BALL_RADII) + 1):
        # an index reaches its own tensors (row e of B^T), then crosses r - 1 more indices: row e of B^T R_(r-1)
        index_reach = B.T @ reach                                     # (I, T)
        reach = reach @ step                                          # (T, T) = R_r
        if r in BALL_RADII:
            tensor_sizes.append(reach.count_nonzero(axis=1))
            index_sizes.append(index_reach.count_nonzero(axis=1))
    tensor_sizes = np.array(tensor_sizes, dtype=float).reshape(len(BALL_RADII), c.n).T          # (T, 4)
    index_sizes = np.array(index_sizes, dtype=float).reshape(len(BALL_RADII), len(c.indices)).T  # (I, 4)
    return np.log2(tensor_sizes) / BALL_SCALE, np.log2(index_sizes) / BALL_SCALE


# === 3. the input =======================================================================

# encoding -> its number of columns. x_tensor and x_index hold the native columns, then every encoding in this
# order; a model reads the native ones and the encodings it asks for (columns()).
ENCODINGS: dict[str, int] = {"rwpe": len(RWPE_STEPS), "balls": len(BALL_RADII)}


def columns(kind: str, encodings: tuple[str, ...]) -> list[int]:
    """The columns of x_<kind> ("tensor" or "index") that a model with these encodings reads: native, then each."""
    start = len(NATIVE_FEATURES[kind])
    out = list(range(start))
    for name, width in ENCODINGS.items():
        if name in encodings:
            out.extend(range(start, start + width))
        start += width
    return out


def _native_columns(c: ConnectedComponent, kind: str) -> np.ndarray:
    """The NATIVE_FEATURES[kind] columns of the tensors or the indices of c, each divided by its scale: (n, F)."""
    columns = np.array([f.function(c) for f in NATIVE_FEATURES[kind]], dtype=np.float32)    # (F, n)
    return columns.T / np.array([f.scale for f in NATIVE_FEATURES[kind]], dtype=np.float32)   # (n, F)


def component_input(c: ConnectedComponent) -> ComponentInput:
    """The input of one component, what the features file holds: native columns, then every encoding (ENCODINGS)."""
    coo = c.incidence_matrix.tocoo()          # B as its list of nonzeros: coo.row[k], coo.col[k] = tensor, index
    edge = np.array([coo.row, coo.col], dtype=np.int64)                                 # (2, E)
    p_tensor, p_index = rwpe(c)
    b_tensor, b_index = ball_sizes(c)
    x_tensor = np.concatenate([_native_columns(c, "tensor"), p_tensor, b_tensor], axis=1).astype(np.float32)
    x_index = np.concatenate([_native_columns(c, "index"), p_index, b_index], axis=1).astype(np.float32)
    return ComponentInput(c.id, x_tensor, x_index, edge)


# === 4. the features file ===============================================================

# A hash of what component_input computes: a features file's meta records it, and a file made with another is
# refused. It sees names, scales and constants, not the body of a function.
FEATURES_ID = hashlib.sha256(repr(([(kind, [(f.name, f.scale) for f in rows])
                                    for kind, rows in NATIVE_FEATURES.items()],
                                   RWPE_STEPS, RWPE_SCALE, RWPE_FLOOR, RWPE_MAX_TENSORS, BALL_RADII, BALL_SCALE)
                             ).encode()).hexdigest()[:16]


def save(path: Path, inputs: list[ComponentInput]) -> None:
    """results/features/<name>.npz: three arrays per input, "<id>/x_tensor", "<id>/x_index" and "<id>/edge"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{x.id}/{field}": getattr(x, field) for x in inputs for field in ("x_tensor", "x_index", "edge")}
    np.savez(path, **arrays)


def load(path: Path) -> dict[str, ComponentInput]:
    """{input.id: input} of save(). FileNotFoundError (naming script 03) without the file; ValueError if its meta
    has another FEATURES_ID."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run scripts/03_compute_features.py")
    config.load_meta(path, "Run scripts/03_compute_features.py with --fresh.", features_id=FEATURES_ID)
    with np.load(path, allow_pickle=False) as data:
        return {key: ComponentInput(key, *(data[f"{key}/{field}"] for field in ("x_tensor", "x_index", "edge")))
                for key in dict.fromkeys(name.split("/")[0] for name in data.files)}


# === 5. the path to the model ==========================================================

# The descriptor rows fall into three groups:
#   given      the size group: exact and cheap, it describes the input itself (the readout receives log n_tensors;
#              a baseline regresses on the sizes)
#   left out   alpha_bisection and its error: not learnt for now (the fit sits at a bound on most components)
#   predicted  every other row: the TARGETS
GIVEN = descriptors.ROWS["n_tensors"].rows
LEFT_OUT = ("alpha_bisection", "alpha_bisection_err")

# How each target is learnt, its kind in TARGETS:
#   "binary"  yes or no: the model gives a logit, the loss is binary cross-entropy, the readout a sigmoid
#   "linear"  of order one or signed (exponents, dimensions, rates, curvatures, a fraction): standardised as it is
#   "log"     >= 0 over orders of magnitude (treewidths, widths, diameters): log1p, then standardised, so that the
#             loss weighs relative errors
# A linear or log target is learnt with a Huber loss on its standardised value (train.Scaler).
BINARY = ("planar", "ball_exponential", "walk_exponential")
LINEAR = ("ball_dimension", "walk_dimension", "ball_growth_rate", "walk_decay_rate", "forman_mean", "forman_min",
          "hyperedge_fraction")
TARGETS: dict[str, str] = {name: "binary" if name in BINARY else "linear" if name in LINEAR else "log"
                           for name in descriptors.ROWS if name not in GIVEN and name not in LEFT_OUT}

# The targets whose probe is slow: the ones a learned predictor is for (the fast ones can simply be computed)
SLOW_TARGETS = tuple(name for name in TARGETS if not descriptors.ROWS[name].fast)


SPLITS = ("random", "family", "size")      # how split_labels cuts: by hash, a family or a member held out, by size
SIZE_SPLIT_MAX_N = 256                     # split "size": train / val up to this n, test above (classifier: 384-1024)


def split_labels(records: list[dict[str, Any]], how: str = "random", held_out: str | None = None) -> list[str]:
    """"train", "val" or "test" for each record of an instances file, in order.

    "random": 80 / 10 / 10 by hash. "family": test is every instance whose family or member (instances.member_name)
    is `held_out` ("geometric": both dims; "geometric-dim3-r_scale1.4": that member); the rest 90 / 10 by hash.
    "size": test is n > SIZE_SPLIT_MAX_N; the rest 90 / 10 by hash. The hash is of the GRAPH (family, knobs, n, seed;
    the name of a benchmark network), so instances that differ only in legs or bond rule fall on one side. val chooses
    when to stop and the settings; test is looked at once, at the end. ValueError for an unknown split or a held_out
    no record has."""
    if how not in SPLITS:
        raise ValueError(f"unknown split {how!r}; valid: {list(SPLITS)}")
    members = [member_name(rec["family"], rec["params"].get("knobs", {})) for rec in records]
    if how == "family":
        if held_out is None:
            raise ValueError("split 'family' needs held_out: the family or member to leave out")
        families = {rec["family"] for rec in records}
        if held_out not in families and held_out not in members:
            raise ValueError(f"held_out {held_out!r} is no family or member of these records: families "
                             f"{sorted(families)}, members {sorted(set(members))}")
    labels = []
    for rec, member in zip(records, members):
        params = rec["params"]
        graph = (rec["family"], sorted(params.get("knobs", {}).items()), params.get("n"), rec["seed"],
                 params.get("name"))
        bucket = int(hashlib.sha256(repr(graph).encode()).hexdigest(), 16) % 10   # 0..9: one graph, one bucket
        if how == "random":
            labels.append("test" if bucket == 0 else "val" if bucket == 1 else "train")
        elif how == "family":
            held = held_out in (rec["family"], member)
            labels.append("test" if held else "val" if bucket == 0 else "train")
        else:
            labels.append("test" if params["n"] > SIZE_SPLIT_MAX_N else "val" if bucket == 0 else "train")
    return labels


def samples(records: list[dict[str, Any]], components_by_instance: dict[str, list[tuple[str, int]]],
            descriptors_by_component: dict[str, dict[str, Any]], features_by_component: dict[str, ComponentInput],
            target_names: tuple[str, ...]) -> list[tuple[ComponentInput, np.ndarray]]:
    """One (ComponentInput, y) per connected component of each record, read from the three files; no graph is built.

    The components of a record come from the components file (02), the input from the features file (03), y (K,)
    float32 in the order of `target_names` from the descriptors file (02), NaN where a row is None. A component above
    RWPE_MAX_TENSORS has no input and is skipped; anything else missing is a ValueError naming the script to run.
    Two instances with the same core (they differ in their legs) give two equal samples."""
    out = []
    for rec in records:
        if rec["id"] not in components_by_instance:
            raise ValueError(f"no components of the instance {rec['id']}: run scripts/02_compute_descriptors.py")
        for position, (component_id, n) in enumerate(components_by_instance[rec["id"]]):
            if n > RWPE_MAX_TENSORS:
                continue
            where = f"component {position} of instance {rec['id']}"
            if component_id not in descriptors_by_component:
                raise ValueError(f"no descriptors of {where}: run scripts/02_compute_descriptors.py")
            if component_id not in features_by_component:
                raise ValueError(f"no features of {where}: run scripts/03_compute_features.py")
            row = descriptors_by_component[component_id]
            y = np.array([np.nan if row.get(name) is None else float(row[name]) for name in target_names],
                         dtype=np.float32)
            out.append((features_by_component[component_id], y))
    return out


def unseen(pairs: list[tuple[ComponentInput, np.ndarray]],
           train_pairs: list[tuple[ComponentInput, np.ndarray]]) -> list[tuple[ComponentInput, np.ndarray]]:
    """The pairs whose component id is not among the train pairs'. split_labels splits graphs, but two graphs can
    share a component (two seeds of a small tree): one met in training measures nothing held out."""
    seen = {x.id for x, _ in train_pairs}
    return [(x, y) for x, y in pairs if x.id not in seen]

"""How one network is made. docs/instances.md.

A network is the triple `Network` = (inputs, output, size_dict): one tuple of index labels per tensor, the open
labels, label -> dimension. Contraction paths refer to it by positions in `inputs`. An `Instance` is the triple plus
where it comes from (family, params, seed); a generated one is rebuilt bit for bit from those, so only structure is
stored, never arrays. What is computed from a network (descriptors, paths, costs) lives in other modules, keyed by id.

Reading order:
1. graph <-> network: tn_to_graph (lossless), graph_to_tn (its inverse), simplify, remove_legs;
2. the families (networkx graphs, and our hyperedge family _hyper), their knobs, the random streams, draw_connected,
   the bond rules, the open legs, the checks;
3. Instance, and generate(): the one pipeline from (family, n, seed, ...) to an Instance, using section 2 in order.

The Einsum Benchmark networks are read by benchmark.py; suite.py says which instances make the dataset.
"""
from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import cotengra as ctg
import networkx as nx
import numpy as np

Inputs = list[tuple[str, ...]]        # inputs[v]: the labels of tensor v, in the order of its axes
Output = tuple[str, ...]              # the open labels, in the order of the result's axes
SizeDict = dict[str, int]             # label -> dimension
ContractionPath = list[tuple[int, int]]   # n-1 pairs of positions in the current list of tensors
Network = tuple[Inputs, Output, SizeDict]  # the triple: everything a contraction needs but the arrays

CHI = 2                               # the default bond dimension of a generated network


# === 1. graph <-> network ==================================================
#
# The graph G(N) of a network, a MultiGraph (two tensors can share several bonds):
#   - a tensor is a vertex 0..n-1, in the order of `inputs`;
#   - a bond (an index on two tensors) is an edge, key = its label;
#   - a trace (an index written twice in one tensor) is a self-loop;
#   - a hyperedge (an index on three or more tensors) or an open leg is a vertex ("index", e), joined to every
#     tensor that writes it;
#   - an index written once, on one tensor, and not open (summed inside that tensor) is the tensor's attribute.
# In networkx, G.nodes[x] is the dict of attributes of node x and G.edges[u, v, key] that of an edge:
#   edge        {"weight": log2 d}                        every edge
#   tensor      {"local": [labels], "local_weight": [log2 d]}   only if it has such indices, else {}
#   index node  {"open": False}                           a hyperedge
#               {"open": True, "pos": i}                  an open leg, i = its axis in the output

def _carriers(inputs: Inputs) -> dict[str, dict[int, int]]:
    """Index label -> {tensor: how many times that tensor writes it}."""
    carriers: dict[str, dict[int, int]] = {}
    for v, term in enumerate(inputs):
        for e in term:
            if e not in carriers:
                carriers[e] = {}
            carriers[e][v] = carriers[e].get(v, 0) + 1
    return carriers


def _add_index_vertex(G: nx.MultiGraph, e: str, writes: dict[int, int], weight: float,
                      output: Output) -> None:
    """Add the vertex ("index", e) of a hyperedge or open leg e, with one edge per time a tensor writes e.

    An edge joins exactly two tensors, so an index on three or more (a hyperedge: a COPY tensor) or one that leaves
    the network (an open leg) becomes a vertex of its own instead."""
    name = ("index", e)                    # one object, a tuple: never mistaken for a tensor, an integer 0..n-1
    G.add_node(name, open=e in output)     # open: is e one of the output's labels?
    if e in output:
        # the output has an order (einsum "ab->ba" is a transpose): keep the axis, for graph_to_tn
        G.nodes[name]["pos"] = output.index(e)
    for v, times in writes.items():
        for j in range(times):
            # parallel edges are told apart by their key. A tensor that writes e twice (T[h,h,x], h also on other
            # tensors) needs TWO edges to the vertex; with one key networkx would keep one, so the second is (e, 1)
            key = e if j == 0 else (e, j)
            G.add_edge(v, name, key=key, weight=weight)


def tn_to_graph(inputs: Inputs, output: Output, size_dict: SizeDict) -> nx.MultiGraph:
    """The lossless graph G(N) of a network (conventions above). graph_to_tn(G) gives the network back, up to the
    names of the labels and the order of the axes inside a tensor. A tensor with no index is an isolated vertex."""
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(inputs)))
    for e, writes in _carriers(inputs).items():
        weight = float(np.log2(size_dict[e]))
        places = sum(writes.values())               # how often e is written in all
        if e in output or places > 2:               # an open leg or a hyperedge: a vertex
            _add_index_vertex(G, e, writes, weight, output)
        elif places == 2:                           # a bond, or a trace
            tensors = list(writes)                  # two tensors for a bond, one for a trace (T[a,a,b]):
            if len(tensors) == 1:                   # then both ends are that tensor, a self-loop
                u = v = tensors[0]
            else:
                u, v = tensors
            G.add_edge(u, v, key=e, weight=weight)
        else:
            # written once and not open (b in sum_{b,c} T[a,b,c] M[c,d]): no second tensor, so no edge; kept on the
            # tensor so that graph_to_tn can write it back. writes = {tensor: 1}; (v,) = writes takes its one key
            (v,) = writes
            # setdefault returns the list under "local", creating it empty the first time
            G.nodes[v].setdefault("local", []).append(e)
            G.nodes[v].setdefault("local_weight", []).append(weight)
    return G


def simplify(G: nx.MultiGraph) -> nx.Graph:
    """The simple graph of a tn_to_graph graph; every vertex and its attributes stay, the labels (keys) go.

    Parallel bonds become one edge whose weight is the sum: two bonds act as one of dimension d1 * d2, exact for cost.
    Self-loops go: a trace is summed inside its tensor before any contraction. A tensor that writes an index vertex
    several times (T[h,h,x]: keys h and (h, 1)) keeps one edge of weight log2 d, not m log2 d: the repeats are a
    diagonal of the one COPY leg."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, weight in G.edges(data="weight", default=1.0):
        if u == v:
            continue
        if H.has_edge(u, v):
            if _is_index_vertex(G, u) or _is_index_vertex(G, v):
                continue                    # a repeat of the same index: already counted
            H[u][v]["weight"] += weight
        else:
            H.add_edge(u, v, weight=weight)
    return H


def remove_legs(G: nx.Graph) -> nx.Graph:
    """A copy of G without the open legs on one tensor (open index vertices of degree 1; the mirror of
    _attach_legs). An open index on two or more tensors stays: it is a hyperedge of the network."""
    H = G.copy()
    H.remove_nodes_from([v for v in G if _is_index_vertex(G, v) and G.nodes[v]["open"] and G.degree(v) == 1])
    return H


def _is_index_vertex(G: nx.Graph, v: Any) -> bool:
    """True for the vertices tn_to_graph makes for a hyperedge or an open leg."""
    return isinstance(v, tuple) and len(v) == 2 and v[0] == "index" and "open" in G.nodes[v]


def graph_to_tn(G: nx.Graph, chi: int = CHI) -> Network:
    """Any networkx graph -> a network: the inverse of tn_to_graph, and how the networkx families become networks.

    One tensor per vertex, one index per edge (a self-loop is a trace), of dimension 2**weight, or chi on an edge
    without a weight. A vertex ("index", e) is one index on all its neighbours; with open=True it goes to the output at
    its "pos". G.nodes[v]["local"] adds indices local to v. A plain graph gives a closed network: the output is ()
    (graph_to_tn(nx.grid_2d_graph(2, 2)) is the 2 x 2 example of docs/instances.md). Labels are a, b, c, ... in
    edge order."""
    G = nx.MultiGraph(G)                    # every edge now has a key (0 in a simple graph)
    size_dict: SizeDict = {}

    def new(weight: float | None) -> str:
        """The next free label a, b, c, ...; its dimension is 2**weight, or chi without a weight."""
        label = ctg.get_symbol(len(size_dict))
        size_dict[label] = chi if weight is None else round(2.0 ** weight)
        return label

    # Pass 1: one label per index. A hyperedge or leg is named once, by its vertex; a bond or a trace by its edge
    # (its two ends and its key).
    name: dict[Any, str] = {}
    for u, v, key, weight in G.edges(keys=True, data="weight"):
        if _is_index_vertex(G, u) or _is_index_vertex(G, v):
            vertex = u if _is_index_vertex(G, u) else v
            if vertex not in name:
                name[vertex] = new(weight)
        else:
            name[frozenset((u, v)), key] = new(weight)

    # Pass 2: the tensors, in node order. The labels of one tensor: one per edge at it (a trace twice: G.edges(t)
    # lists a self-loop once), then its local indices.
    inputs: Inputs = []
    for t in (v for v in G.nodes if not _is_index_vertex(G, v)):
        term: list[str] = []
        for _, other, key in G.edges(t, keys=True):
            label = name[other] if _is_index_vertex(G, other) else name[frozenset((t, other)), key]
            term += [label, label] if other == t else [label]
        weights = G.nodes[t].get("local_weight", [])
        for j in range(len(G.nodes[t].get("local", []))):
            term.append(new(weights[j] if j < len(weights) else None))
        inputs.append(tuple(term))

    legs = [v for v in G.nodes if _is_index_vertex(G, v) and G.nodes[v]["open"]]
    legs.sort(key=lambda v: G.nodes[v].get("pos", 0))                  # no "pos": keep the node order
    return inputs, tuple(name[v] for v in legs), size_dict


# === 2. the families (docs/instances.md) =====================================================

def side(n: int, dim: int) -> int:
    """The side L of a dim-dimensional lattice of n = L**dim tensors; ValueError for any other n."""
    L = round(n ** (1 / dim))
    if L**dim != n:
        raise ValueError(f"a {dim}D lattice has L**{dim} tensors; n = {n} is not of that form "
                         f"(nearest: {L**dim})")
    return L


def _smallworld(n: int, seed: int, degree: int, p: float) -> nx.Graph:
    """Newman-Watts: a ring joining each tensor to its `degree` nearest neighbours, plus shortcuts
    that are only added. The ring stays, so the graph is connected by construction (Watts-Strogatz
    rewires ring bonds and can cut the ring)."""
    return nx.newman_watts_strogatz_graph(n, degree, p, seed=seed)


# family -> f(n, seed, **knobs) -> nx.Graph; the knobs are the family's KNOBS (below), all given by the caller, so
# no function hides a default. `lambda n, s: ...` is a one-line function, `def f(n, s): return ...`.
#   lattice     tensors on a dim-dimensional grid, bonds between nearest neighbours.
#   randreg     every tensor has exactly `degree` bonds, partners paired at random: no geometry.
#   geometric   n points uniform in the unit dim-cube, a bond between two points closer than r = r_scale / n**(1/dim):
#               r shrinks with n, so the mean degree stays about fixed.
#   smallworld  a ring plus random shortcuts (_smallworld).
#   tree        a uniformly random labelled tree.
# At a fixed mean degree a geometric graph falls apart more often as n grows, so a suite's r_scale is chosen for few
# redraws (draw_connected). Most redraws over seeds 0-9, n = 16..1024: dim 2, r_scale 1.8: 3 (1.6: 22 at n = 1024);
# dim 3, r_scale 1.4: 6, mean degree 6-9 (1.6: 2, but mean degree 11-14).
FAMILIES: dict[str, Callable[..., nx.Graph]] = {
    "lattice":     lambda n, s, dim: nx.grid_graph(dim=[side(n, dim)] * dim),
    "randreg":     lambda n, s, degree: nx.random_regular_graph(degree, n, seed=s),
    "geometric":   lambda n, s, dim, r_scale: nx.random_geometric_graph(n, r_scale / n ** (1 / dim),
                                                                        dim=dim, seed=s),
    "smallworld":  _smallworld,
    "tree":        lambda n, s: nx.random_labeled_tree(n, seed=s),
}


def _unused_labels(size_dict: SizeDict, count: int) -> list[str]:
    """`count` index names that size_dict does not use: the next symbols after its own."""
    labels: list[str] = []
    j = len(size_dict)
    while len(labels) < count:
        label = ctg.get_symbol(j)
        j += 1
        if label not in size_dict:
            labels.append(label)
    return labels


def _hyper(n: int, seed: int, degree: int, hyper_share: float, rank: int, chi: int) -> Network:
    """Our hyperedge family: a random `degree`-regular network (chi per bond) plus round(hyper_share * n) new indices
    of dimension chi, each on `rank` distinct tensors drawn uniformly. Closed: the legs come from LEGS, as for every
    family. Not cotengra's rand_equation: there the rank of a hyperedge is random in 3..n, here it is a knob."""
    inputs, output, size_dict = graph_to_tn(nx.random_regular_graph(degree, n, seed=seed), chi)
    inputs = [tuple(term) for term in inputs]
    rng = np.random.default_rng([seed, 1])          # the hyperedges' own stream, apart from the graph's
    new = round(hyper_share * n)
    for label in _unused_labels(size_dict, new):
        size_dict[label] = chi
        for t in rng.choice(n, size=rank, replace=False):     # `rank` different tensors
            inputs[int(t)] += (label,)
    return inputs, output, size_dict


HYPER = "hyper"                  # not a graph: a regular graph plus hyperedges (_hyper)


# --- the knobs ----------------------------------------------------------------------

# family -> the names of its knobs, the arguments of its draw, in the order params["knobs"] stores them. generate
# needs every one; their values are chosen in suite.py.
#   lattice     dim: the dimension of the grid (n = L**dim).
#   randreg     degree: bonds per tensor.
#   geometric   dim: the dimension of the box; r_scale: the bond radius in units of n**(-1/dim).
#   smallworld  degree: nearest ring neighbours each tensor is joined to; p: shortcut probability.
#   hyper       degree: bonds per tensor of the regular graph underneath; hyper_share: new hyperedge
#               indices as a share of n; rank: tensors each hyperedge is written on.
KNOBS: dict[str, tuple[str, ...]] = {
    "lattice": ("dim",), "tree": (),
    "randreg": ("degree",),
    "geometric": ("dim", "r_scale"),
    "smallworld": ("degree", "p"),
    HYPER: ("degree", "hyper_share", "rank"),
}


# --- the streams and the draw ----------------------------------------------------------

# One stored seed, three independent random streams, SeedSequence([seed, stream, attempt]): the graph, the bond
# dimensions, the legs. Changing the bond rule or the legs cannot change the graph.
STREAM_GRAPH, STREAM_BONDS, STREAM_LEGS = 0, 1, 2


def _stream(seed: int, stream: int, attempt: int = 0) -> np.random.SeedSequence:
    return np.random.SeedSequence([seed, stream, attempt])


def _graph_seed(seed: int, attempt: int) -> int:
    """The integer seed of draw attempt `attempt`: it decides the graph (or the hyper network)."""
    return int(_stream(seed, STREAM_GRAPH, attempt).generate_state(1)[0])


def _bonds_rng(seed: int) -> np.random.Generator:
    """The generator that decides the bond dimensions (spread_bonds)."""
    return np.random.default_rng(_stream(seed, STREAM_BONDS))


def _legs_rng(seed: int) -> np.random.Generator:
    """The generator that decides which tensors carry the open legs (the LEGS patterns)."""
    return np.random.default_rng(_stream(seed, STREAM_LEGS))


# How each family comes to be connected, stored as Instance.params["connected_by"]:
#   "construction": every draw is connected, so attempt is 0 (a lattice, a tree, the ring);
#   "conditioning": a draw can be disconnected, and draw_connected redraws until it is not.
CONNECTED_BY: dict[str, str] = {
    "lattice": "construction", "tree": "construction",
    "smallworld": "construction",
    "geometric": "conditioning",
    "randreg": "conditioning", HYPER: "conditioning",
}
CONNECT_TRIES = 100              # draw_connected gives up after this many attempts


def draw_connected(draw: Callable[[int], Network], seed: int,
                   tries: int = CONNECT_TRIES) -> tuple[Inputs, Output, SizeDict, int]:
    """(inputs, output, size_dict, attempt): the first connected network that draw(s) gives, attempt a drawing from
    the seed of (seed, STREAM_GRAPH, a). ValueError after `tries` attempts.

    Keeping the first connected draw is an exact draw from the family conditioned on being connected: attempts are
    independent, and every connected draw is accepted alike. Why connected: a closed network in k pieces is a product
    of k networks, which teaches the scheduler nothing and puts lambda_2 = 0 in the descriptors."""
    for attempt in range(tries):
        inputs, output, size_dict = draw(_graph_seed(seed, attempt))
        if nx.is_connected(simplify(tn_to_graph(inputs, output, size_dict))):
            return inputs, output, size_dict, attempt
    raise ValueError(f"no connected draw in {tries} tries: the family's knob is in the wrong regime")


def _draw_function(family: str, n: int, chi: int, knobs: dict[str, Any]) -> Callable[[int], Network]:
    """draw(s) -> one network of the family from the integer seed s, for draw_connected."""
    if family == HYPER:
        def draw_hyper(s: int) -> Network:
            return _hyper(n, s, chi=chi, **knobs)
        return draw_hyper

    def draw_graph(s: int) -> Network:
        return graph_to_tn(FAMILIES[family](n, s, **knobs), chi)
    return draw_graph


# --- the bond rule ---------------------------------------------------------------------

BOND_RULES = ("uniform", "spread")


def spread_bonds(size_dict: SizeDict, rng: np.random.Generator,
                 log2_range: tuple[int, int] = (1, 4)) -> SizeDict:
    """The bond rule "spread": log2 d of every index drawn uniformly from log2_range, ends included (d = 2..16).

    With one chi the cost depends on the structure alone; with spread dimensions a few large contractions dominate,
    and an order that ignores sizes goes wrong (the Einsum Benchmark has such spreads). generate passes
    _bonds_rng(seed): the graph stays the one of the uniform rule."""
    low, high = log2_range
    exponents = rng.integers(low, high + 1, size=len(size_dict)).tolist()
    return {e: 2 ** x for e, x in zip(sorted(size_dict), exponents)}


# --- the open legs: a new index on ONE tensor and in the output ---------------------------

def _legs_none(inputs: Inputs, k: int, rng: np.random.Generator) -> list[int]:
    return []


def _legs_scattered(inputs: Inputs, k: int, rng: np.random.Generator) -> list[int]:
    """k tensors drawn uniformly with replacement: a tensor may get several legs."""
    return [int(v) for v in rng.integers(0, len(inputs), size=k)]


def _tensor_neighbours(inputs: Inputs) -> dict[int, set[int]]:
    """Tensor -> the tensors that share an index with it."""
    neighbours: dict[int, set[int]] = {v: set() for v in range(len(inputs))}
    for writes in _carriers(inputs).values():
        for v in writes:
            neighbours[v].update(writes)
    for v, others in neighbours.items():
        others.discard(v)
    return neighbours


def _legs_region(inputs: Inputs, k: int, rng: np.random.Generator) -> list[int]:
    """The k tensors nearest to a random tensor: a connected block, as for a reduced density matrix. Ties at the last
    distance are broken at random. k tensors are always reached: the network is connected and k <= n."""
    G = nx.Graph(_tensor_neighbours(inputs))       # tensor graph: an edge where two tensors share an index
    start = int(rng.integers(len(inputs)))          # 1. the centre: one tensor, uniformly
    ball: list[int] = []
    for layer in nx.bfs_layers(G, start):           # 2. layer d = the tensors at distance d from it
        if len(ball) == k:
            break
        # 3. take as many as are still needed, in a random order; sorted first, so the order networkx lists the
        #    layer in cannot change the draw
        shuffled = [int(v) for v in rng.permutation(sorted(layer))]
        ball += shuffled[: k - len(ball)]
    return ball


def _legs_mps(inputs: Inputs, k: int, rng: np.random.Generator) -> list[int]:
    """One leg per tensor: a state, as in MPS / PEPS."""
    return list(range(len(inputs)))


# pattern -> f(inputs, k, rng) -> the tensor that carries each leg, one entry per leg
LEGS: dict[str, Callable[..., list[int]]] = {
    "closed": _legs_none, "scattered": _legs_scattered, "region": _legs_region, "mps": _legs_mps,
}
LEGS_WITH_K = ("scattered", "region")      # the patterns whose k the caller chooses (closed has 0, mps n)


def _attach_legs(network: Network, tensors: list[int], d_open: int) -> Network:
    """The network with one new open leg per entry of `tensors` (a tensor listed twice gets two): a fresh label of
    dimension d_open on that tensor, appended to the output. A leg joins no second tensor: connectivity stays."""
    inputs, output, size_dict = network
    new_inputs = [tuple(term) for term in inputs]
    new_sizes = dict(size_dict)
    new_output = list(output)
    for tensor, label in zip(tensors, _unused_labels(size_dict, len(tensors))):
        new_sizes[label] = d_open
        new_inputs[tensor] += (label,)
        new_output.append(label)
    return new_inputs, tuple(new_output), new_sizes


# --- the checks: every refusal of generate() is here --------------------------------------

def _is_int(x: Any) -> bool:
    """An integer (numpy's included), and not a bool: True is not a size."""
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_real(x: Any) -> bool:
    """A finite real number (numpy's included), not a bool, not nan or inf."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def _check_n(family: str, n: Any, minimum: int = 1) -> None:
    """n is an int, and at least `minimum` (every family needs 1; a tree needs 2: one bond)."""
    if not _is_int(n) or n < minimum:
        raise ValueError(f"n must be an int >= {minimum} for {family}, not {n!r}")


def _check_dim(dim: Any) -> None:
    if not _is_int(dim) or dim < 1:
        raise ValueError(f"dim must be an int >= 1, not {dim!r}")


def _check_lattice(family: str, n: int, dim: int) -> None:
    _check_n(family, n)
    _check_dim(dim)
    side(n, dim)


def _check_tree(family: str, n: int) -> None:
    _check_n(family, n, minimum=2)


def _check_randreg(family: str, n: int, degree: int) -> None:
    _check_n(family, n)
    if not _is_int(degree) or degree < 3:
        raise ValueError(f"degree must be an int >= 3, not {degree!r}")
    if n <= degree:
        raise ValueError(f"a random {degree}-regular graph needs n > degree (distinct neighbours); n = {n}")
    if n * degree % 2:
        raise ValueError(f"a random {degree}-regular graph needs n * degree even (n * degree / 2 "
                         f"bonds); n = {n}")


def _check_geometric(family: str, n: int, dim: int, r_scale: float) -> None:
    _check_n(family, n)
    _check_dim(dim)
    if not _is_real(r_scale) or r_scale <= 0:
        raise ValueError(f"r_scale must be a finite real number > 0, not {r_scale!r}")


def _check_smallworld(family: str, n: int, degree: int, p: float) -> None:
    _check_n(family, n)
    if not _is_int(degree) or degree < 1:
        raise ValueError(f"degree must be an int >= 1, not {degree!r}")
    if n <= degree:
        raise ValueError(f"a Newman-Watts ring joins each tensor to its degree = {degree} nearest "
                         f"neighbours, which needs n > degree; n = {n}")
    if not _is_real(p) or not 0 <= p <= 1:
        raise ValueError(f"p must be a real number in [0, 1], not {p!r}")


def _check_hyper(family: str, n: int, degree: int, hyper_share: float, rank: int) -> None:
    _check_randreg(family, n, degree)
    if not _is_real(hyper_share) or hyper_share < 0:
        raise ValueError(f"hyper_share must be a finite real number >= 0, not {hyper_share!r}")
    if not _is_int(rank) or not 3 <= rank <= n:
        raise ValueError(f"rank must be an int in 3..n = {n} (tensors per hyperedge), not {rank!r}")


# family -> check(family, n, **knobs) of n and the knob values, before any draw: a size the family cannot have fails
# with its reason, not with networkx's error.
CHECKS: dict[str, Callable[..., None]] = {
    "lattice": _check_lattice, "tree": _check_tree,
    "randreg": _check_randreg, "geometric": _check_geometric,
    "smallworld": _check_smallworld, HYPER: _check_hyper,
}


def _check_legs(legs: str, k: int, n: int) -> None:
    """ValueError for a legs pattern and a number of legs that make no sense together."""
    if not isinstance(legs, str) or legs not in LEGS:
        raise ValueError(f"legs must be one of {list(LEGS)}, not {legs!r}")
    if legs == "closed" and k != 0:
        raise ValueError(f"legs='closed' has no legs, but k = {k}")
    if legs == "mps" and k not in (0, n):               # k = 0 is the default: it means n
        raise ValueError(f"legs='mps' puts one leg on each of the n = {n} tensors, but k = {k}")
    if legs in LEGS_WITH_K and k < 1:
        raise ValueError(f"legs={legs!r} needs k >= 1, not k = {k}")
    if legs == "region" and k > n:
        raise ValueError(f"legs='region' needs k <= n = {n}, not k = {k}")


def _check_args(family: str, n: int, seed: int, chi: int, bonds: str, d_open: int | None,
                legs: str, k: int, knobs: dict[str, Any]) -> None:
    """Every refusal of generate(), before anything is drawn: the family, seed (int >= 0), chi and d_open (ints >= 2;
    d_open None means chi), the bond rule, k (int), the knob names (exactly KNOBS[family]), then n and the knob values
    (CHECKS), then legs with k and n (_check_legs)."""
    if not isinstance(family, str) or family not in KNOBS:
        raise ValueError(f"unknown family {family!r}; known: {list(KNOBS)}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be an int >= 0, not {seed!r}")
    if not _is_int(chi) or chi < 2:
        raise ValueError(f"chi must be an int >= 2 (the dimension of a bond), not {chi!r}")
    if bonds not in BOND_RULES:
        raise ValueError(f"bonds must be one of {BOND_RULES}, not {bonds!r}")
    if d_open is not None and (not _is_int(d_open) or d_open < 2):
        raise ValueError(f"d_open must be an int >= 2, not {d_open!r}")
    if not _is_int(k):
        raise ValueError(f"k must be an int, not {k!r}")
    unknown = [name for name in knobs if name not in KNOBS[family]]
    if unknown:
        raise ValueError(f"unknown knob {unknown} for {family}; its knobs: {list(KNOBS[family])}")
    missing = [name for name in KNOBS[family] if name not in knobs]
    if missing:
        raise ValueError(f"{family} needs the knob(s) {missing}: pass every one of {list(KNOBS[family])}")
    CHECKS[family](family, n, **knobs)
    _check_legs(legs, k, n)


# === 3. the record =======================================================================

def network_id(inputs: Inputs, output: Output, size_dict: SizeDict) -> str:
    """16 hex digits of sha256(einsum string, sizes in label order): the same network, the same hash."""
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    blob = eq + "|" + ",".join(f"{e}:{size_dict[e]}" for e in sorted(size_dict))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def member_name(family: str, knobs: dict[str, Any]) -> str:
    """The class of an instance in tables: "benchmark", else the family and its knobs ("lattice-dim3")."""
    return "benchmark" if family == "benchmark" else "-".join([family, *(f"{k}{v}" for k, v in knobs.items())])


# @dataclass writes __init__, __repr__ and __eq__ from the annotated fields, in their order:
# Instance(inputs, output, size_dict, family, params, seed). frozen=True forbids reassigning a field, so the cached
# properties computed from the fields stay true.
@dataclass(frozen=True)
class Instance:
    """One network and where it came from; nothing derived is stored.

    `family` is provenance, not a label: it serves the held-out-class split, per-class tables and the rebuild.
    == compares every field, provenance included; `id` compares the network alone."""
    inputs: Inputs
    output: Output
    size_dict: SizeDict
    family: str               # a key of FAMILIES, "hyper", or "benchmark"
    params: dict[str, Any]    # generated: n, chi, bonds, knobs (all, in KNOBS order), attempt, connected_by,
                              # legs, k, d_open, log2_output, log2_total; benchmark: n, name, category, paths
    seed: int | None          # None for the benchmark

    # computed on demand: a property on every access, a cached_property once, on the first
    @property
    def network(self) -> Network:
        """The triple: what contraction paths refer to (positions in `inputs`)."""
        return self.inputs, self.output, self.size_dict

    @property
    def name(self) -> str:
        """A readable name, unique within a suite: the family, n, every knob, chi, the bond rule, the legs and the
        seed."""
        p = self.params
        if self.family == "benchmark":
            return f"benchmark-{p['name']}"
        knobs = "".join(f"-{k}{v}" for k, v in p["knobs"].items())
        legs = "" if p["legs"] == "closed" else f"-{p['legs']}{p['k']}-d{p['d_open']}"
        return f"{self.family}-n{p['n']}{knobs}-chi{p['chi']}-{p['bonds']}{legs}-s{self.seed}"

    @cached_property
    def eq(self) -> str:
        """The network as an einsum string, e.g. "ab,bc->ac": what the file stores instead of the inputs."""
        return ctg.utils.inputs_output_to_eq(self.inputs, self.output)

    @cached_property
    def graph(self) -> nx.MultiGraph:
        """G(N), the lossless graph (tn_to_graph); never saved."""
        return tn_to_graph(self.inputs, self.output, self.size_dict)

    @cached_property
    def core(self) -> nx.Graph:
        """H: G(N) simple (simplify) and without its open legs (remove_legs), the only graph the descriptors read.
        Without legs the class of a network does not depend on them. With bonds only, H is tensors and bonds;
        hyperedges stay as index vertices."""
        return remove_legs(simplify(self.graph))

    @cached_property
    def components(self) -> int:
        """The number of connected pieces (of the core; legs change nothing): 1 for every generated instance."""
        return nx.number_connected_components(self.core)

    @cached_property
    def member(self) -> str:
        """The class of the instance in tables: member_name of its family and knobs."""
        return member_name(self.family, self.params.get("knobs", {}))

    @cached_property
    def id(self) -> str:
        """network_id of the triple: the same network, the same id, whatever its provenance."""
        return network_id(self.inputs, self.output, self.size_dict)


def _log2_sizes(output: Output, size_dict: SizeDict) -> tuple[float, float]:
    """(log2_output, log2_total): log2 of the entries of the output tensor; the sum of log2 d over ALL indices.

    log2_total bounds every path without a search: a step involves at most all the indices, so every intermediate has
    at most 2**log2_total entries (W <= log2_total) and a path costs at most (n - 1) 2**log2_total. With
    log2_total < 1000 and n < 2**24 no cost overflows a float64."""
    log2_output = float(sum(np.log2(size_dict[e]) for e in output))
    log2_total = float(sum(np.log2(d) for d in size_dict.values()))
    return log2_output, log2_total


def generate(family: str, n: int, seed: int, chi: int = CHI, bonds: str = "uniform",
             legs: str = "closed", k: int = 0, d_open: int | None = None, **knobs: Any) -> Instance:
    """One instance, bit for bit from (family, n, seed, chi, bond rule, legs, k, d_open, knobs).

    The steps, in order:
      1. _check_args: a ValueError with its reason for any bad argument, before anything is drawn. `knobs` are
         every knob of the family (KNOBS);
      2. draw the closed network (_hyper, or the family's graph through graph_to_tn, chi per bond) until it is
         connected (draw_connected);
      3. bonds="spread": new dimensions (spread_bonds), same structure;
      4. legs: the LEGS pattern picks the k tensors that carry an open leg ("mps": all n), each of dimension d_open
         (default chi). How many legs is the suite's choice (suite._legs_k);
      5. params["log2_output"], params["log2_total"] (_log2_sizes).
    No cost is formed here: whoever forms one checks log2_total against 1023 first."""
    _check_args(family, n, seed, chi, bonds, d_open, legs, k, knobs)
    d_open = chi if d_open is None else d_open
    knobs = {name: knobs[name] for name in KNOBS[family]}      # the stored order, whatever the caller's
    draw = _draw_function(family, n, chi, knobs)

    inputs, output, size_dict, attempt = draw_connected(draw, seed)
    if bonds == "spread":
        size_dict = spread_bonds(size_dict, _bonds_rng(seed))
    k = n if legs == "mps" else k                              # mps: one leg per tensor, whatever k was
    carriers = LEGS[legs](inputs, k, _legs_rng(seed))
    inputs, output, size_dict = _attach_legs((inputs, output, size_dict), carriers, d_open)
    log2_output, log2_total = _log2_sizes(output, size_dict)

    params = {"n": len(inputs), "chi": chi, "bonds": bonds, "knobs": knobs, "attempt": attempt,
              "connected_by": CONNECTED_BY[family], "legs": legs, "k": k, "d_open": d_open,
              "log2_output": log2_output, "log2_total": log2_total}
    return Instance(inputs, output, size_dict, family, params, seed)

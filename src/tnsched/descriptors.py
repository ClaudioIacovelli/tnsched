"""Structural descriptors of a network: numbers and flags read off its graph. docs/descriptors.md.

They make "which optimizer wins" a statement about a class of networks, not about a list of files. Every descriptor
is measured on ONE connected component of H = inst.core (the network simplified, without open legs). Words: a tensor
is a vertex 0..n-1; an index is a bond or a hyperedge; w_e = log2 d_e is its weight in bits. Four views of a
component:
  H            bonds are edges, a hyperedge is a vertex ("index", e). The graph of the treewidth, topology and
               curvature rows.
  incidence    tensors and indices as vertices, an edge between a tensor and each of its indices (with bonds only, H
               with every edge subdivided). The graph for counting steps: a distance between tensors is half its own.
  index graph  one vertex per index, an edge when two indices share a tensor (with bonds only, the line graph of H).
               Its weighted treewidth is T*, the log2 cost of the largest step of the best contraction tree [MS08].
  cut graph    tensors only, every index of r tensors a clique on them (the primal graph), weighted
               w_e / (floor(r/2) ceil(r/2)) per pair; with bonds only, H. The graph of the cut rows.
The cut of a set S of tensors, cut_bits(S), is the sum of w_e over the indices with tensors on both sides of S, each
once: it defines W (log2 of the largest intermediate) and b (the balanced cut). No cut of the cut graph exceeds it
(cut_graph says why), so the cut graph gives lower bounds on W and is where halve() searches.

Reading order:
1. ConnectedComponent and its views, split(), the three primitives (tensor_distances, cut_bits, walk), the spectrum;
2. one function per descriptor, f(component) -> number | bool | None, grouped as in docs/descriptors.md. None means
   not defined, never inf or nan: rows go to JSON. The docstring is one line: what the number is, its docs section
   and source keys;
3. the table DESCRIPTORS (its columns are explained above it) and table();
4. measure(component), save, load.

Left out: exact carving width of planar networks (ratcatcher), hyperbolicity, modularity, Ollivier curvature (an
optimal transport per edge).
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar

from . import config
from .instances import graph_to_tn, network_id

# lambda_2 of H (the spectral term of tw_lower): dense up to 300 vertices, then an iterative solver. A random
# 4-regular graph is the slow case: 1.4 s at 3 000 tensors, 2.8 s at 4 096, 11 s at 6 000
SPECTRAL_MAX_N = 4096
# the rows on the cut graph: lambda_2_cut and b_lower take 0.3 s at 32 000 pairs of tensors; b_upper (halve) about
# 4e-5 s per pair, 2 s at 50 000 pairs (5 s at 32 418 pairs and 17 000 tensors)
CUT_GRAPH_MAX_PAIRS = 50_000
# the dimension rows: with fewer points a power law and an exponential cannot be told apart. A ball has at most
# ecc/2 radii by nature, a walk up to WALK_MAX_STEPS steps, hence two numbers
BALL_MIN_POINTS = 4              # the ball rows: radii 1 .. ecc/2
WALK_MIN_POINTS = 6              # the walk rows: steps 4 .. the last with p >= MIX_FACTOR pi
# p_k - pi is fitted on a log scale, where a tiny difference is noise: stop while it is still >= 3 pi (p >= 4 pi)
MIX_FACTOR = 4
WALK_MAX_STEPS = 4096            # an expander stops after a few steps; a ring or a large lattice never mixes in range
# the separator exponent alpha of b(m) = a + scale m^alpha is searched in ALPHA_BOUNDS (a separator grows at most
# like m). Within BOUND_TOL of a bound the minimum is the edge of the search, not a stationary point: no error bar
ALPHA_BOUNDS = (0.01, 1)
BOUND_TOL = 1e-3

Value = float | int | bool | None


# H holds tensors (ints) and index vertices (tuples): every pass over H has to tell them apart
def _is_index(v: Any) -> bool:
    """An index vertex ("index", e) of H as opposed to a tensor (an int)."""
    return isinstance(v, tuple)


# === 1. the component, the three primitives, the spectrum ===============================

@dataclass(frozen=True, eq=False)
class ConnectedComponent:
    """One connected component H of inst.core, with its views (module docstring) built on first use. `seed` feeds
    every random choice of a probe, so measure() is deterministic."""
    H: nx.Graph
    seed: int = 0
    # results keyed by arguments: the spectra of gap(), by graph and weighting
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # every descriptor and every view assumes this, so it is checked where the object is built
        hint = "one connected component: build it with split()"
        if not self.tensors:
            raise ValueError(f"H has no tensor; {hint}")
        if not nx.is_connected(self.H):
            raise ValueError(f"H is not connected; {hint}")
        if any(w <= 0 for _, _, w in self.H.edges(data="weight", default=1.0)):
            raise ValueError(f"every weight (log2 d) must be > 0; {hint}")

    # cached_property: what the component IS, computed once: the tensors, the views, structural facts
    @cached_property
    def tensors(self) -> tuple[int, ...]:
        return tuple(sorted(v for v in self.H if not _is_index(v)))

    @property
    def n(self) -> int:
        return len(self.tensors)

    @cached_property
    def degrees(self) -> tuple[int, ...]:
        """The number of indices on each tensor, in the order of self.tensors."""
        return tuple(self.incidence_graph.degree(t) for t in self.tensors)

    @cached_property
    def incidence_graph(self) -> nx.Graph:
        """The incidence graph: the tensors, then the indices in the order of H's edges; every edge carries the w of
        its index. A bond a-b becomes the index vertex ("bond", a, b)."""
        G = nx.Graph()
        G.add_nodes_from(self.tensors)
        for u, v, w in self.H.edges(data="weight", default=1.0):
            # a hyperedge vertex keeps its name; a bond becomes the vertex ("bond", a, b), a < b
            index = u if _is_index(u) else v if _is_index(v) else ("bond", min(u, v), max(u, v))
            # the index joins its tensor ends: two for a bond, one per edge of a hyperedge vertex
            G.add_weighted_edges_from((t, index, w) for t in (u, v) if not _is_index(t))
        return G

    @cached_property
    def incidence_matrix(self) -> sparse.csr_array:
        """B (T, I), B[t, e] = 1 when index e is on tensor t; rows in the order of self.tensors, columns of
        self.indices (weight=None: incidences, not bits)."""
        B = nx.bipartite.biadjacency_matrix(self.incidence_graph, self.tensors, self.indices, weight=None)
        return sparse.csr_array(B)

    @cached_property
    def indices(self) -> tuple:
        """The bonds and hyperedges: the vertices of the incidence graph that are not tensors."""
        return tuple(v for v in self.incidence_graph if _is_index(v))

    @cached_property
    def w(self) -> dict:
        """index -> w_e = log2 d_e (bits)."""
        # every edge of an index carries its w_e: read the first
        return {k: next(iter(self.incidence_graph[k].values()))["weight"] for k in self.indices}

    @cached_property
    def index_graph(self) -> nx.Graph:
        """The index graph: the incidence graph projected on its indices (nodes in the order of self.indices)."""
        return nx.bipartite.projected_graph(self.incidence_graph, self.indices)

    @cached_property
    def incidence_girth(self) -> int | None:
        """The girth of the incidence graph (twice the fewest indices on a cycle); None without a cycle."""
        g = nx.girth(self.incidence_graph)
        return None if g == math.inf else int(g)            # nx.girth is inf on a forest; rows hold None

    @cached_property
    def forman_per_edge(self) -> list[float]:
        """F(e) = 4 - deg(u) - deg(v) + 3 (triangles through e) for every edge u-v of H, the Forman curvature that
        forman_mean and forman_min summarise."""
        H = self.H
        # triangles through u-v: the neighbours of one end that also neighbour the other. Walking the end of smaller
        # degree makes an edge cost its smaller degree (an edge at a hub stays cheap)
        values = []
        for u, v in H.edges:
            small, large = (u, v) if H.degree(u) <= H.degree(v) else (v, u)
            values.append(4 - H.degree(u) - H.degree(v) + 3 * sum(x in H[large] for x in H[small]))
        return values

    @cached_property
    def ranks(self) -> dict:
        """index -> r_e, the number of tensors it joins."""
        return {k: self.incidence_graph.degree(k) for k in self.indices}

    @cached_property
    def id(self) -> str:
        """network_id of H turned back into a network (graph_to_tn: a weight w is a dimension 2**w). For a connected,
        closed network of single bonds only it is the instance's id."""
        return network_id(*graph_to_tn(self.H))

    @cached_property
    def cut_graph(self) -> nx.Graph:
        """The cut graph: the primal graph (tensors only, every index of r tensors a clique on them) with weight
        w_e / (floor(r/2) ceil(r/2)) per pair; with bonds only, H itself.

        A split k | r - k cuts k (r - k) <= floor(r/2) ceil(r/2) pairs, so the clique adds at most w_e, the index's
        share of cut_bits(S): the cut of this graph never exceeds cut_bits(S). An index of r tensors has r (r - 1) / 2
        pairs: measure() skips the rows on this graph above CUT_GRAPH_MAX_PAIRS."""
        if not any(map(_is_index, self.H)):              # bonds only: H already is the cut graph
            return self.H.copy()
        G = nx.Graph()
        G.add_nodes_from(self.tensors)
        for k in self.indices:
            r = self.ranks[k]
            # sorted: the edges go in in a fixed order
            for a, b in itertools.combinations(sorted(self.incidence_graph[k]), 2):
                # two indices on the same pair of tensors: their weights add
                old = G[a][b]["weight"] if G.has_edge(a, b) else 0.0
                G.add_edge(a, b, weight=old + self.w[k] / ((r // 2) * (r - r // 2)))
        return G


def split(H: nx.Graph, seed: int, tensors: Any = None) -> list[ConnectedComponent]:
    """The connected pieces of H, or of these tensors of H, largest first (ties: smallest tensor).
    docs/descriptors.md."""
    if tensors is not None:
        keep = set(tensors)
        # a bond stays when both its tensors do (subgraph); a hyperedge vertex with one tensor joins nothing
        keep |= {x for x in H if _is_index(x) and sum(t in keep for t in H[x]) >= 2}
        H = H.subgraph(keep)
    parts = [ConnectedComponent(H.subgraph(p).copy(), seed) for p in nx.connected_components(H)]
    return sorted(parts, key=lambda P: (-len(P.H), P.tensors[0]))


# --- the three primitives: tensor_distances, cut_bits, walk -----------------------------

def tensor_distances(c: ConnectedComponent, s: int) -> dict[int, int]:
    """tensor -> the number of indices crossed from s (a hyperedge counts 1, whatever its rank): half the BFS
    distance on the incidence graph. docs/descriptors.md, views of a component."""
    d = nx.single_source_shortest_path_length(c.incidence_graph, s)
    # tensor -> index -> tensor is 2 edges, for a bond and a hyperedge alike
    return {t: k // 2 for t, k in d.items() if not _is_index(t)}


def cut_bits(c: ConnectedComponent, A: Any) -> float:
    """w(dA) in bits, log2 of the entries of the intermediate that contracts the tensors A: the sum of w_e over the
    indices with tensors in A and outside it, each once. docs/descriptors.md, views of a component."""
    A, done, total = set(A), set(), 0.0
    for u, v, w in c.H.edges(data="weight", default=1.0):       # every edge of a hyperedge carries its w_e
        if _is_index(u) or _is_index(v):
            x = u if _is_index(u) else v
            # a hyperedge has one edge per tensor in H: count it at its first edge
            if x in done:
                continue
            done.add(x)
            ends = list(c.H[x])
        else:
            ends = [u, v]
        if 0 < sum(t in A for t in ends) < len(ends):           # 0 < inside < r: tensors on both sides
            total += w
    return total


def walk(c: ConnectedComponent) -> sparse.csr_array:
    """P (T, T) = D_T^-1 B D_I^-1 B^T, the random walk on the tensors: from t one of its indices uniformly, then one
    tensor of that index uniformly [ZHS06]. With bonds only, the lazy walk (1 + D^-1 A) / 2.
    docs/descriptors.md, estimates."""
    B = c.incidence_matrix                                     # (T, I), B[t, e] = 1 if index e is on tensor t
    # d_T: the indices on each tensor (its degree); d_I: the tensors of each index (its rank)
    d_T, d_I = np.asarray(B.sum(axis=1)).ravel(), np.asarray(B.sum(axis=0)).ravel()
    # D_T^-1 B (T, I): tensor -> each of its indices with probability 1/d_T; D_I^-1 B^T (I, T): index -> each of its
    # tensors with probability 1/d_I. The product P[t, u] = Prob(t -> u in two steps); rows sum to 1
    return (sparse.diags(1.0 / d_T) @ B @ sparse.diags(1.0 / d_I) @ B.T).tocsr()


# --- the spectrum: _laplacian_spectrum, gap ---------------------------------------------

# one solver for every gap, so lambda_2 of H and of the cut graph are computed alike
def _laplacian_spectrum(G: nx.Graph, weight: str | None) -> tuple[float, np.ndarray]:
    """(lambda_2, Fiedler vector in the order of G): dense up to 300 vertices, else networkx's sparse
    tracemin solver (seeded). The graph is connected (a ConnectedComponent and its cut graph are); one
    vertex has lambda_2 = 0 and a zero vector."""
    nodes = list(G)
    if len(nodes) < 2:
        return 0.0, np.zeros(len(nodes))
    if len(nodes) <= 300:
        values, vectors = np.linalg.eigh(nx.laplacian_matrix(G, nodelist=nodes, weight=weight).toarray().astype(float))
        return float(values[1]), vectors[:, 1]          # ascending: [0] is the zero mode of a connected graph
    lam = nx.algebraic_connectivity(G, weight=weight, method="tracemin_lu", seed=0)
    vec = nx.fiedler_vector(G, weight=weight, method="tracemin_lu", seed=0)
    return float(lam), np.asarray(vec)


def gap(c: ConnectedComponent, which: str, weight: str | None) -> tuple[float, np.ndarray]:
    """(lambda_2, Fiedler vector) of the Laplacian of c.H or c.cut_graph (`which` = "H" | "cut";
    weight None = every edge 1), kept on the component."""
    key = (which, weight)                                             # one spectrum per graph and weighting
    if key not in c._cache:
        G = c.H if which == "H" else c.cut_graph                     # the cut graph is built only when asked for
        c._cache[key] = _laplacian_spectrum(G, weight)
    return c._cache[key]


# === 2. the descriptors, in the order of docs/descriptors.md =====================================

# --- size ---

def size(c: ConnectedComponent) -> tuple[int, int, int, Value, Value]:
    """(n_tensors, n_indices, n_incidences, log2_d_mean, log2_d_std): how big the component is. docs/descriptors.md, size."""
    bits = list(c.w.values())                          # w_e = log2 d_e of every index, in bits
    # an index is a bond or a hyperedge; the incidences (sum of r_e) are the edges of the incidence graph
    return (c.n, len(c.indices), c.incidence_graph.number_of_edges(),
            float(np.mean(bits)) if bits else None, float(np.std(bits)) if bits else None)


# --- degree ---

def degree_mean(c: ConnectedComponent) -> Value:
    """The mean number of indices on a tensor. docs/descriptors.md, degree."""
    return float(np.mean(c.degrees))


def degree_max(c: ConnectedComponent) -> int:
    """The largest number of indices on a tensor: the Delta of the bound tw -> T. docs/descriptors.md, degree."""
    return max(c.degrees)


def n_hyperedges(c: ConnectedComponent) -> int:
    """Indices on 3 or more tensors, or open on 2: the index vertices of H. docs/descriptors.md, degree."""
    return sum(1 for v in c.H if _is_index(v))


def hyperedge_fraction(c: ConnectedComponent) -> float:
    """Hyperedges over indices. docs/descriptors.md, degree."""
    return n_hyperedges(c) / len(c.indices) if c.indices else 0.0


def hyperedge_size_max(c: ConnectedComponent) -> int:
    """The largest r_e: 2 on a network of bonds, 0 without an index. docs/descriptors.md, degree."""
    return max(c.ranks.values(), default=0)


# --- distance ---

def diameter(c: ConnectedComponent) -> Value:
    """The most indices crossed between two tensors; None for one tensor. docs/descriptors.md, distance."""
    if c.n < 2:
        return None
    return max(max(tensor_distances(c, s).values()) for s in c.tensors)   # one BFS per tensor, hence the cap


def diameter_lower(c: ConnectedComponent) -> Value:
    """A lower bound on the diameter by 4 double sweeps from a seeded tensor. docs/descriptors.md, distance [MLH09]."""
    if c.n < 2:
        return None
    rng = np.random.default_rng(c.seed)
    v = c.tensors[int(rng.integers(c.n))]
    best = 0
    for _ in range(4):                                            # each sweep starts from the last farthest
        dist = tensor_distances(c, v)
        far = max(dist.values())
        v = min(u for u, d in dist.items() if d == far)          # the farthest, smallest label on ties
        best = max(best, far)
    return best


def girth(c: ConnectedComponent) -> Value:
    """The fewest indices crossed by a cycle, half the incidence girth; None without one. docs/descriptors.md, distance."""
    g = c.incidence_girth
    return None if g is None else g // 2


# --- dimension ---
# Two growths: the balls |B(r)| and the return p_k - pi of the walk. A power law (a lattice) has a dimension D; an
# exponential (an expander, a regular tree, a hyperbolic graph) has a rate. Both laws are fitted on the same window,
# the smaller residual wins (a tie: power). The balls vote, one fit per source; the walk is one fit on the average.
# Each fit returns its three rows: the flag "exponential", the power law's D, the exponential's rate (one is None).

def _fit_both(x: np.ndarray, y: np.ndarray, min_points: int) -> tuple[str | None, float | None]:
    """(regime, slope): ("power", slope of y on log x) or ("exponential", slope of y on x), whichever
    fits better (a tie: power); (None, None) below min_points points."""
    if len(x) < min_points:                        # too few points to tell a law
        return None, None
    power = np.polyfit(np.log(x), y, 1)            # power law: straight in log x
    exponential = np.polyfit(x, y, 1)              # exponential: straight in x
    rss_power = float(np.sum((np.polyval(power, np.log(x)) - y) ** 2))
    rss_exp = float(np.sum((np.polyval(exponential, x) - y) ** 2))
    if rss_exp < rss_power:
        return "exponential", float(exponential[0])
    return "power", float(power[0])


def ball_fit(c: ConnectedComponent) -> tuple[Value, Value, Value]:
    """(ball_exponential, ball_dimension, ball_growth_rate): the growth of the balls of tensors. docs/descriptors.md, dimension."""
    # the law of log |B(r)| per source (<= 8; r = 1 .. ecc/2), the majority of the sources with a window
    # (a tie: power), and the mean slope of that law: D (power) or c (exponential); all None without a window
    rng = np.random.default_rng(c.seed)
    slopes: dict[str, list[float]] = {"power": [], "exponential": []}
    for i in rng.choice(c.n, size=min(8, c.n), replace=False):   # up to 8 centres: average out the local shape
        dist = tensor_distances(c, c.tensors[int(i)])
        rmax = max(dist.values()) // 2                 # half the eccentricity: farther, the ball feels the edge
        cumulative = np.cumsum(np.bincount(list(dist.values())))  # cumulative[r] = |B(r)|, within r indices
        r = np.arange(1, rmax + 1, dtype=float)
        regime, slope = _fit_both(r, np.log(cumulative[1:rmax + 1]), BALL_MIN_POINTS)
        if regime is not None:                         # a short window does not vote
            slopes[regime].append(slope)
    power, exponential = slopes["power"], slopes["exponential"]
    if not (power or exponential):
        return None, None, None
    regime = "exponential" if len(exponential) > len(power) else "power"
    value = float(np.mean(slopes[regime]))             # |B(r)| ~ r^D (slope on log r) or e^(c r) (slope on r)
    if regime == "power":
        return False, value, None
    return True, None, value


def _return_probability_mean(c: ConnectedComponent) -> tuple[np.ndarray, float]:
    """(p, pi): p[k - 1] the probability that a walker is back at its start after k steps (walk()), averaged over
    <= 32 starts; pi its limit, the stationary probability of those starts. Stops at the first p < MIX_FACTOR pi or
    after WALK_MAX_STEPS. Needs 2 tensors."""
    n = c.n
    P = walk(c)                                        # P[t, s]: probability of the step t -> s
    PT = P.T.tocsr()                                   # P^T: one product moves every walker one step
    degree = np.array(c.degrees, dtype=float)          # indices per tensor
    rng = np.random.default_rng(c.seed)
    sources = rng.choice(n, size=min(32, n), replace=False)   # up to 32 starts: average out the local shape
    columns = np.arange(len(sources))
    X = np.zeros((n, len(sources)))                    # column j: where walker j is, as probabilities
    X[sources, columns] = 1.0                          # walker j starts on sources[j] with certainty
    pi = degree[sources].mean() / degree.sum()         # where p ends: the stationary probability
    p = []
    for _ in range(WALK_MAX_STEPS):
        X = PT @ X                                     # every walker takes one step
        p.append(X[sources, columns].mean())           # back at its own start, averaged
        if p[-1] < MIX_FACTOR * pi:                    # mixed: the rest is noise on a log scale
            break
    return np.array(p), pi


def walk_fit(c: ConnectedComponent) -> tuple[Value, Value, Value]:
    """(walk_exponential, walk_dimension, walk_decay_rate): the return probability of the walk. docs/descriptors.md, dimension [ZHS06]."""
    # the law of p_k - pi on the window k >= 4 with p_k >= MIX_FACTOR pi: d_s = -2 x the slope on log k
    # (power) or gamma = -the slope on k (exponential); all None with fewer than WALK_MIN_POINTS steps
    if c.n < 2:
        return None, None, None                        # no step to take
    p, pi = _return_probability_mean(c)
    k = np.arange(1, len(p) + 1, dtype=float)          # the step numbers of p
    keep = (k >= 4) & (p >= MIX_FACTOR * pi)           # past the first steps (lattice detail), not yet mixed
    # a power law in k, or an exponential decay: a gap
    regime, slope = _fit_both(k[keep], np.log(p[keep] - pi), WALK_MIN_POINTS)
    if regime == "power":
        return False, float(-2 * slope), None          # p_k - pi ~ k^(-D/2), so D = -2 x slope
    if regime == "exponential":
        return True, None, float(-slope)               # p_k - pi ~ e^(-gamma k), so gamma = -slope
    return None, None, None


# --- treewidth and weighted treewidth: two brackets, four rows ---------------------------
# tw(H), unweighted (the same as the incidence graph's):                          tw_lower <= tw(H) <= tw_upper
# T* = tw_w(L), the weighted treewidth of the index graph L in bits: the largest step of the best tree [MS08]:
#                                                                                 tw_w_lower <= T* <= tw_w_upper
# A lower row is the max of its methods, each a valid bound, with the reason above it.

# every subgraph of a graph of treewidth k has a vertex of degree <= k: the largest min degree
def _degeneracy(G: nx.Graph) -> int:
    """The largest core number of G: a lower bound on tw(G)."""
    return max(nx.core_number(G).values())


# ints and tuples do not compare: a total order makes the heap's ties, hence the value, reproducible
def _order(v: Any) -> tuple:
    """A total order on vertices of H or of the index graph: tensors by number, then the tuples."""
    return (1, v) if _is_index(v) else (0, v)


# the same for every minor; minor-min-width, a heuristic value of the contraction degeneracy [BKW06]
def _mmd(G: nx.Graph) -> int:
    """Minor-min-width of G, least-common-neighbour rule: a lower bound on tw(G)."""
    adj = {v: set(G[v]) for v in G}             # the current minor of G: G with the contractions done so far
    heap = [(len(adj[v]), _order(v), v) for v in adj]              # smallest degree first, ties by _order
    heapq.heapify(heap)
    best = 0
    while len(adj) > 1:
        degree, _, v = heapq.heappop(heap)
        # a stale entry: the heap cannot update or remove an entry, so a vertex whose degree changed gets a new one
        # and a removed vertex keeps its old ones; an entry is used only if it still describes adj
        if v not in adj or len(adj[v]) != degree:
            continue
        best = max(best, degree)                                   # the min degree of every minor bounds tw
        neighbours = adj.pop(v)
        if neighbours:                                             # an isolated vertex is only deleted
            for w in neighbours:
                adj[w].discard(v)
            # contract v into the neighbour with the fewest neighbours in common (ties by _order)
            u = min(neighbours, key=lambda w: (len(neighbours & adj[w]), _order(w)))
            gained = neighbours - {u}
            adj[u] |= gained
            for w in gained:
                adj[w].add(u)
            for w in neighbours:
                heapq.heappush(heap, (len(adj[w]), _order(w), w))  # their degrees changed
    return best


# ceil(n lambda_2 / (Delta + lambda_2) - 1), n, Delta, lambda_2 of H, any graph with an edge [GHN24]
def _spectral_tw(c: ConnectedComponent) -> int:
    """The spectral lower bound on tw(H); 0 (still valid) on one vertex or above SPECTRAL_MAX_N tensors."""
    if len(c.H) < 2 or c.n > SPECTRAL_MAX_N:
        return 0
    lam = gap(c, "H", None)[0]                                     # lambda_2 of H, every edge 1
    top = max(d for _, d in c.H.degree())                          # Delta
    # - 1e-9: an exact integer that comes out a hair above is not rounded up
    return max(0, math.ceil(len(c.H) * lam / (top + lam) - 1 - 1e-9))


def tw_lower(c: ConnectedComponent) -> int:
    """The max of three lower bounds on tw(H): degeneracy, minor-min-width, spectral. docs/descriptors.md, treewidth [BKW06, GHN24]."""
    return max(_degeneracy(c.H), _mmd(c.H), _spectral_tw(c))


def tw_upper(c: ConnectedComponent) -> int:
    """The width of the min-fill elimination order of H. docs/descriptors.md, treewidth."""
    # any elimination order is a tree decomposition: its width bounds tw from above
    return int(nx.approximation.treewidth_min_fill_in(c.H)[0])     # (width, tree decomposition)


def tw_w_lower(c: ConnectedComponent) -> float:
    """A lower bound on T* in bits: the max of the heaviest tensor, a bag of tw(L) + 1 indices, w_min tw_lower. docs/descriptors.md, weighted treewidth [BKW06, GHN24]."""
    if not c.indices:
        return 0.0
    w_min = min(c.w.values())
    # the indices of a tensor are a clique of the index graph L, so one bag holds them all
    largest_tensor = max(sum(c.w[k] for k in c.incidence_graph[t]) for t in c.tensors)
    L = c.index_graph
    # a bag of k indices weighs >= k w_min, and the best decomposition of L has a bag of tw(L) + 1 of them
    bag = w_min * (1 + max(_degeneracy(L), _mmd(L)))
    # tw(L) + 1 >= tw(H)
    return float(max(largest_tensor, bag, w_min * tw_lower(c)))


def tw_w_upper(c: ConnectedComponent) -> float:
    """An upper bound on T* (and on W) in bits: the heaviest bag of the min-fill order of the index graph. docs/descriptors.md, weighted treewidth [MS08]."""
    if not c.indices:
        return 0.0
    # an elimination order of the index graph IS a contraction order (summing an index = contracting its
    # tensors), every step inside its bag: the largest bag in bits bounds T* from above, and W too (W <= T)
    bags = nx.approximation.treewidth_min_fill_in(c.index_graph)[1].nodes   # each bag a frozenset of indices
    return float(max(sum(c.w[x] for x in bag) for bag in bags))


# --- expansion, width and bisection (W, b) ---

# the spectral gap of H (expander or not) and of the cut graph (the bound on W)
def lambda_2(c: ConnectedComponent) -> Value:
    """The algebraic connectivity of H, every edge 1: expander or not. docs/descriptors.md, expansion."""
    return gap(c, "H", None)[0] if len(c.H) > 1 else None


def lambda_2_cut(c: ConnectedComponent) -> Value:
    """The weighted algebraic connectivity of the cut graph. docs/descriptors.md, width."""
    return gap(c, "cut", "weight")[0] if c.n > 1 else None


def b_lower(c: ConnectedComponent) -> Value:
    """2 lambda_2_cut n / 9 bits: a lower bound on every balanced cut (n/3 <= |S| <= 2n/3), so on b and on W, since cut(S) >= lambda_2 |S| (n - |S|) / n. docs/descriptors.md, width [Moh92]."""
    lam = lambda_2_cut(c)
    return None if lam is None else 2 * lam * c.n / 9


def halve(c: ConnectedComponent, starts: int = 5, seed: int | None = None) -> tuple[float, set, set]:
    """(cut in bits, A, B): the best of 1 + `starts` halvings of the tensors, measured by cut_bits, so an upper
    bound. The starts are the Fiedler vector of the cut graph cut at its median and `starts` random halves, each
    refined by Kernighan-Lin on the cut graph. Needs 2 tensors."""
    rng = np.random.default_rng(c.seed if seed is None else seed)
    G = c.cut_graph
    nodes = list(G)
    half = len(nodes) // 2
    # the Fiedler start: the half lowest on the weighted Fiedler vector, kept by gap() (stable: ties keep node order)
    order = np.argsort(gap(c, "cut", "weight")[1], kind="stable")
    starts_from = [{nodes[i] for i in order[:half]}]
    for _ in range(starts):
        starts_from.append({nodes[i] for i in rng.permutation(len(nodes))[:half]})
    best: tuple[float, set, set] | None = None
    for A in starts_from:
        B = set(nodes) - A
        A, B = nx.community.kernighan_lin_bisection(G, partition=(A, B), weight="weight",
                                                    seed=int(rng.integers(2**31)))   # KL's own seed, from rng
        value = cut_bits(c, A)                                     # the true cut, not the cut graph's
        if best is None or value < best[0]:
            best = (value, set(A), set(B))
    assert best is not None                                        # for the type checker: starts_from is not empty
    return best


def b_upper(c: ConnectedComponent) -> Value:
    """The cut of halve()'s best split: an upper bound on the balanced bisection, bits. docs/descriptors.md, width."""
    # its own call: alpha's cap is lower, so the bisection profile is not always there to read it from
    return halve(c)[0] if c.n > 1 else None


def bisection_profile(c: ConnectedComponent, min_size: int = 4) -> tuple[list[int], list[float]]:
    """(sizes, cuts): recursive halving, the size m (tensors) and cut b(m) of every piece of >= 2 min_size tensors.
    A half that falls apart goes on as its connected pieces (a cut of 0 between them: no point of b(m))."""
    sizes: list[int] = []
    cuts: list[float] = []
    stack = [c]                                                    # the pieces still to cut
    while stack:
        P = stack.pop()
        if P.n < 2 * min_size:
            continue
        value, A, B = halve(P, seed=c.seed + len(sizes))           # one seed per cut, all fixed by c.seed
        sizes.append(P.n)
        cuts.append(value)
        stack.extend(split(P.H, c.seed, A))                        # the connected pieces of each half wait to be cut
        stack.extend(split(P.H, c.seed, B))
    return sizes, cuts


def alpha(c: ConnectedComponent) -> tuple[Value, Value]:
    """(alpha_bisection, alpha_bisection_err): the exponent of b(m) = a + scale m^alpha fitted on bisection_profile, and its error. docs/descriptors.md, separator exponent."""
    sizes, cuts = bisection_profile(c)
    if len(set(sizes)) < 3:
        return None, None                                          # a, c and alpha need 3 distinct sizes
    m, b = np.array(sizes, float), np.array(cuts, float)

    def fit(exponent: float) -> tuple[np.ndarray, float]:
        """(a, c) and the residual at a fixed exponent."""
        X = np.stack([np.ones_like(m), m ** exponent], axis=1)
        coef = np.linalg.lstsq(X, b, rcond=None)[0]
        return coef, float(np.sum((X @ coef - b) ** 2))

    # at a fixed alpha the model is linear in (a, scale): optimise alpha alone, (a, scale) by least squares inside
    alpha_hat = float(minimize_scalar(lambda x: fit(x)[1], bounds=ALPHA_BOUNDS, method="bounded").x)
    if len(m) < 4:
        return alpha_hat, None                     # three parameters fit three points exactly: no residual for an error
    if min(alpha_hat - ALPHA_BOUNDS[0], ALPHA_BOUNDS[1] - alpha_hat) < BOUND_TOL:
        return alpha_hat, None                                     # at a bound the minimum is the edge of the search
    (a, scale), residual = fit(alpha_hat)
    # the error of alpha is the least-squares covariance at the optimum: sigma^2 (J^T J)^-1, J the derivatives of
    # a + scale m^alpha with respect to (a, scale, alpha) and sigma^2 = residual / (points - 3)
    J = np.stack([np.ones_like(m), m ** alpha_hat, scale * m ** alpha_hat * np.log(m)], axis=1)
    if np.linalg.matrix_rank(J) < 3:
        return alpha_hat, None                                     # a flat profile (scale = 0): alpha changes nothing
    return alpha_hat, math.sqrt(residual / (len(m) - 3) * np.linalg.inv(J.T @ J)[2, 2])


# --- topology ---

def planar(c: ConnectedComponent) -> bool:
    """Whether H, hence the network (never the primal graph), is planar. docs/descriptors.md, topology."""
    return bool(nx.check_planarity(c.H)[0])


def genus_lower(c: ConnectedComponent) -> int:
    """Euler's bound on the genus with the incidence girth; 0 without a cycle. docs/descriptors.md, topology."""
    g = c.incidence_girth
    if g is None:
        return 0
    # Euler on the incidence graph (n_I vertices, m_I edges): n - m + f = 2 - 2 genus, and every face
    # has length >= g, so f <= 2m / g and genus >= (m (g - 2) - g (n - 2)) / (2 g)
    n_I, m_I = c.n + len(c.indices), c.incidence_graph.number_of_edges()
    numerator = m_I * (g - 2) - g * (n_I - 2)
    return max(0, -(-numerator // (2 * g)))                        # -(-x // y) = ceil(x / y) in integers


# --- curvature ---

def forman_mean(c: ConnectedComponent) -> Value:
    """The mean augmented Forman curvature of the edges of H. docs/descriptors.md, curvature [For03]."""
    values = c.forman_per_edge
    return float(np.mean(values)) if values else None


def forman_min(c: ConnectedComponent) -> Value:
    """The smallest augmented Forman curvature of an edge of H. docs/descriptors.md, curvature [For03]."""
    values = c.forman_per_edge
    return float(min(values)) if values else None


# === 3. the table =======================================================================

class Probe(NamedTuple):
    """One entry of DESCRIPTORS: a function and the rows it fills (several when it returns a tuple)."""
    function: Callable[[ConnectedComponent], Value | tuple[Value, ...]]
    kind: str                               # one of KINDS: the value is exact, a bound, or an estimate
    bounds: str                             # one of BOUNDS: which quantity a bound or estimate is about
    on: str                                 # one of ON: the view it is measured on
    source: str                             # the keys of the README References that define it, or "-"
    fast: bool                              # False: seconds or more
    max_size: int | None                    # above this the rows are None and listed as "skipped"
    unit: str                               # what max_size counts: "tensors" | "indices" | "pairs" |
                                            # "tensors + pairs" | "edges" ("-": no cap)
    complexity: str                         # how the cost grows with the size
    rows: tuple[str, ...]


# the values allowed in the columns kind, bounds and on (the tests check every row). BOUNDS: W space (log2 of the
# largest intermediate, bits), T time (log2 of the largest step, bits), tw treewidth of H, b balanced cut (bits),
# alpha separator exponent, D a dimension, "-" a plain property
KINDS = ("exact", "upper", "lower", "estimate")
BOUNDS = ("W", "T", "tw", "b", "alpha", "D", "-")
ON = ("H", "incidence", "index graph", "cut", "-")

# Caps: a row runs where one call takes about 2 s at most (timed on the benchmark and the suite). max_size counts
# the unit that predicts the row's cost: edges of H for the Forman rows (each edge counts common neighbours), pairs
# of tensors for the cut rows, tensors + pairs (the vertices and edges of the cut graph) for alpha_bisection, which
# halves every piece of every level: 2 s at about 3 500, where tensors alone (4 s at 905 tensors with 48 000 pairs)
# or pairs alone (4 s at 3 000 pairs, 2 000 tensors) mispredict it.
# complexity: n, m = vertices and edges of H (on the cut graph m counts pairs, r_e (r_e - 1) / 2 per index),
# Delta = the largest degree, I = indices, r_e = the tensors of index e. The views, built once, are not counted.
# The Forman rows: 2.5 to 3 microseconds per edge, 2 s at the largest benchmark network (800 000 edges)
FORMAN_MAX_EDGES = 1_000_000
DESCRIPTORS = (
    #     function            kind        bounds on            source         fast   max_size, unit  complexity, rows
    Probe(size,               "exact",    "-",     "-",           "-",           True,  None, "-",
          "O(m)", rows=("n_tensors", "n_indices", "n_incidences", "log2_d_mean", "log2_d_std")),
    Probe(degree_mean,          "exact",    "-",   "-",          "-",            True,  None, "-",
          "O(m)", rows=("degree_mean",)),
    Probe(degree_max,           "exact",    "-",   "-",          "-",            True,  None, "-",
          "O(m)", rows=("degree_max",)),
    Probe(n_hyperedges,         "exact",    "-",   "-",          "-",            True,  None, "-",
          "O(m)", rows=("n_hyperedges",)),
    Probe(hyperedge_fraction,   "exact",    "-",   "-",          "-",            True,  None, "-",
          "O(m)", rows=("hyperedge_fraction",)),
    Probe(hyperedge_size_max,   "exact",    "-",   "-",          "-",            True,  None, "-",
          "O(m)", rows=("hyperedge_size_max",)),
    Probe(diameter,             "exact",    "-",   "incidence",  "-",            False, 1_500, "tensors",
          "O(n m)", rows=("diameter",)),
    Probe(diameter_lower,       "lower",    "-",   "incidence",  "MLH09",        True,  None, "-",
          "O(m)", rows=("diameter_lower",)),
    Probe(girth,                "exact",    "-",   "incidence",  "-",            False, 1_000, "tensors",
          "O((n + m) m)", rows=("girth",)),
    Probe(ball_fit,               "estimate", "D",   "incidence",  "-",            True,  200_000, "tensors",
          "O(m)", rows=("ball_exponential", "ball_dimension", "ball_growth_rate")),
    Probe(walk_fit,             "estimate", "D",   "incidence",  "ZHS06",        True,  10_000, "tensors",
          "O(K s sum r_e^2), K <= WALK_MAX_STEPS, s <= 32", rows=("walk_exponential", "walk_dimension", "walk_decay_rate")),
    Probe(tw_lower,             "lower",    "tw",  "H",          "BKW06, GHN24", True,  150_000, "tensors",
          "O(m log n) (degeneracy, minor-min-width) + the spectral term up to SPECTRAL_MAX_N", rows=("tw_lower",)),
    Probe(tw_upper,             "upper",    "tw",  "H",          "-",            False, 800, "tensors",
          "O(n^2 log n) to O(n^4) worst", rows=("tw_upper",)),
    Probe(tw_w_lower,           "lower",    "T",   "index graph","BKW06, GHN24", True,  25_000, "indices",
          "tw_lower on H + minor-min-width of the index graph, about 6 tensors per index in the largest networks", rows=("tw_w_lower",)),
    Probe(tw_w_upper,           "upper",    "T",   "index graph","MS08",         False, 1_200, "indices",
          "O(I^2 log I) to O(I^4) worst", rows=("tw_w_upper",)),
    Probe(lambda_2,             "exact",    "-",   "H",          "-",            True,  4_096, "tensors",
          "O(n^3) dense (n <= 300) / iterative", rows=("lambda_2",)),
    Probe(lambda_2_cut,         "exact",    "-",   "cut",        "-",            True,  CUT_GRAPH_MAX_PAIRS, "pairs",
          "O(n^3) dense (n <= 300) / iterative", rows=("lambda_2_cut",)),
    Probe(b_lower,              "lower",    "b",   "cut",        "Moh92",        True,  CUT_GRAPH_MAX_PAIRS, "pairs",
          "O(n^3) dense (n <= 300) / iterative", rows=("b_lower",)),
    Probe(b_upper,              "upper",    "b",   "cut",        "-",            False, CUT_GRAPH_MAX_PAIRS // 2, "pairs",
          "6 starts x <= 10 KL passes of O(m log n), + the Fiedler vector of lambda_2_cut", rows=("b_upper",)),
    Probe(alpha,                "estimate", "alpha","cut",        "-",            False, 3_500, "tensors + pairs",
          "O(m log^2 n) (halve on every piece of every level) + their Fiedler vectors", rows=("alpha_bisection", "alpha_bisection_err")),
    Probe(planar,               "exact",    "-",   "H",          "-",            True,  80_000, "tensors",
          "O(n log Delta)", rows=("planar",)),
    Probe(genus_lower,          "lower",    "-",   "incidence",  "-",            False, 1_000, "tensors",
          "O((n + m) m)", rows=("genus_lower",)),
    Probe(forman_mean,          "exact",    "-",   "H",          "For03",        True,  FORMAN_MAX_EDGES, "edges",
          "O(m Delta)", rows=("forman_mean",)),
    Probe(forman_min,           "exact",    "-",   "H",          "For03",        True,  FORMAN_MAX_EDGES, "edges",
          "O(m Delta)", rows=("forman_min",)),
)

# row name (a key of the jsonl) -> its probe
ROWS: dict[str, Probe] = {row: probe for probe in DESCRIPTORS for row in probe.rows}


def table_id(slow: bool = True) -> str:
    """A hash of the rows, caps and units of the probes that run (all, or the fast ones). A descriptors file's meta
    records it, so a file made by another table, or with the fast rows only, is never read as this one."""
    probes = [(p.rows, p.max_size, p.unit) for p in DESCRIPTORS if slow or p.fast]
    return hashlib.sha256(repr(probes).encode()).hexdigest()[:16]


def table() -> str:
    """DESCRIPTORS as a markdown table: the table of docs/descriptors.md is this output."""
    lines = ["| rows | kind | bounds | on | source | complexity | cost |", "|---|---|---|---|---|---|---|"]
    for probe in DESCRIPTORS:
        cap = "" if probe.max_size is None else f", <= {probe.max_size:,} {probe.unit}"
        cost = ("fast" if probe.fast else "slow") + cap
        rows = ", ".join(f"`{r}`" for r in probe.rows)
        lines.append(f"| {rows} | {probe.kind} | {probe.bounds} | {probe.on} | {probe.source} | {probe.complexity} "
                     f"| {cost} |")
    return "\n".join(lines)


# === 4. measure =========================================================================

def _check_measure_args(c: Any, slow: Any) -> None:
    if not isinstance(c, ConnectedComponent):
        raise ValueError(f"c must be a ConnectedComponent, not {type(c).__name__}: build it with split()")
    if not isinstance(slow, bool):
        raise ValueError(f"slow must be a bool, not {slow!r}")


def _size(c: ConnectedComponent, unit: str) -> int:
    """The size of c in `unit`, what a row's max_size counts (Probe.unit)."""
    if unit == "tensors":
        return c.n
    if unit == "indices":
        return len(c.indices)
    if unit == "edges":
        return c.H.number_of_edges()
    pairs = sum(r * (r - 1) // 2 for r in c.ranks.values())      # "pairs" or "tensors + pairs"
    return pairs if unit == "pairs" else c.n + pairs


# rows go to JSON: no numpy scalars, no inf or nan
def _plain(x: Any) -> Value:
    """numpy scalars -> Python numbers; never inf or nan."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    return x if math.isfinite(x) else None


def measure(c: ConnectedComponent, slow: bool = True) -> dict[str, Any]:
    """Every row of DESCRIPTORS on c: one line of results/descriptors/<name>.jsonl.

    {"id": c.id, <row>: value, ..., "skipped": [rows not run], "seconds": {probe: s, "total": s}}. A value is None
    where it is not defined or was skipped. slow=False runs the fast probes only. Deterministic (c.seed)."""
    _check_measure_args(c, slow)
    row: dict[str, Any] = {"id": c.id}
    skipped = []
    seconds: dict[str, float] = {}
    t0 = time.perf_counter()
    for probe in DESCRIPTORS:
        names = probe.rows
        # over its cap, or a slow probe when only the fast ones run: all its rows are None and skipped
        if (probe.max_size is not None and _size(c, probe.unit) > probe.max_size) or (not slow and not probe.fast):
            row.update(dict.fromkeys(names))
            skipped.extend(names)
            continue
        t1 = time.perf_counter()
        values = probe.function(c)
        if len(names) > 1:                                              # a group: one value per row
            row.update(zip(names, map(_plain, values)))
        else:
            row[names[0]] = _plain(values)
        seconds[probe.function.__name__] = time.perf_counter() - t1
    seconds["total"] = time.perf_counter() - t0
    row["skipped"] = skipped
    row["seconds"] = seconds
    return row


# --- the descriptors file: one row per component id --------------------------------------------

def save(path: Path, row: dict[str, Any]) -> None:
    """Append one row (measure()): a long run keeps every row that finished. The meta is written by script 02."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(config.json_line(row) + "\n")


def load(path: Path, slow: bool = True, seed: int = 0) -> dict[str, dict[str, Any]]:
    """{component id: row} of the file. FileNotFoundError (naming script 02) without the file; ValueError if its
    meta is missing or has another table_id or seed."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run scripts/02_compute_descriptors.py")
    config.load_meta(path, "Run scripts/02_compute_descriptors.py with --fresh.", table_id=table_id(slow), seed=seed)
    return {row["id"]: row for row in config.load_lines(path)}


def load_components(path: Path) -> dict[str, list[tuple[str, int]]]:
    """{instance id: [(component id, tensors)]}, largest first (split()), from the components file of script 02.
    FileNotFoundError (naming script 02) without the file."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run scripts/02_compute_descriptors.py")
    return {line["instance"]: [(cid, n) for cid, n in line["components"]] for line in config.load_lines(path)}

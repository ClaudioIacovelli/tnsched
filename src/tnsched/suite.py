"""The dataset: which instances are made, and the file that stores them. docs/suite.md.

instances.py makes one network; this module lists which ones. A `Suite` is a set of members (a family with its knobs
and sizes), seeds, bond rules and open legs; every combination is one instance. SUITES holds two: "reference", the
test set, and "classifier", the training set of the descriptor classifier.

Reading order: Member and Suite; the tables; build() and its steps (plan -> generate -> drop duplicates -> add the
benchmark); the file (record / from_record, save / load).
"""
from __future__ import annotations

import pathlib
from collections.abc import Iterable
from itertools import product              # product(a, b): every pair (x, y), x from a, y from b
from typing import Any, NamedTuple

import cotengra as ctg

from . import benchmark, config
from .instances import CHI, HYPER, LEGS_WITH_K, Instance, generate, network_id


class Member(NamedTuple):
    """One row of a suite: a family with ALL its knobs, and the sizes it is drawn at."""
    family: str                  # a key of instances.KNOBS
    knobs: dict[str, Any]        # every knob of the family, written out
    sizes: tuple[int, ...]       # exact n: lattices need n = L**D
    alpha: float | None = None   # the width W = c * n**alpha bits (see REFERENCE_MEMBERS); needed only for open legs
    c: float | None = None


class Suite(NamedTuple):
    """What one file of instances holds: every (member, size, seed, bond rule, legs, share) is one instance."""
    members: tuple[Member, ...]
    seeds: tuple[int, ...]
    bonds: tuple[str, ...]                 # names of instances.BOND_RULES
    legs: dict[str, tuple[float, ...]]     # legs family (a key of instances.LEGS) -> its shares (see REFERENCE_LEGS)


# The reference suite, the test set: one row per (family, knobs), with its sizes. A new row or size grows the dataset.
#   alpha, c   the width W of cotengra's greedy path on the closed network grows as W = c * n**alpha bits. alpha comes
#              from theory: tree 0; 2D lattice and geometric 1/2; 3D 2/3; random regular, small-world, hyper 1.
#              c is the geometric mean of W / n**alpha over seeds 0-2 at the sizes below, 2 digits. They are frozen:
#              a new c changes the stored file (the slow test of test_suite fails).
#   The fit is loose where theory's alpha meets a small n. Worst ratio of fit to measured W over these sizes and
#   seeds: smallworld x2.6, geometric dim 3 x2.0, geometric dim 2 x1.8, tree x1.5 (W grows from 3 to 7 over
#   n = 10..1024, alpha = 0 says constant), the others x1.1-1.4. So alpha and c only scale the number of open legs;
#   the width itself is measured by a path finder.
#   r_scale of "geometric": the radius factor that needs few redraws to be connected (instances.FAMILIES).
REFERENCE_MEMBERS = (
    Member("lattice",     {"dim": 2},        (16, 64, 256, 1024),     0.5, 1.5),    # L = 4, 8, 16, 32
    Member("lattice",     {"dim": 3},        (27, 64, 216, 1000),     2 / 3, 1.7),  # L = 3, 4, 6, 10
    Member("randreg",     {"degree": 3},     (16, 64, 256, 1024),     1.0, 0.23),
    Member("randreg",     {"degree": 4},     (16, 64, 256, 1024),     1.0, 0.45),
    Member("geometric",   {"dim": 2, "r_scale": 1.8}, (16, 64, 256, 1024), 0.5, 3.5),
    Member("geometric",   {"dim": 3, "r_scale": 1.4}, (16, 64, 256, 1024), 2 / 3, 2.0),
    Member("smallworld",  {"degree": 4, "p": 0.1}, (16, 64, 256, 1024), 1.0, 0.19),
    Member("tree",        {},                (10, 16, 64, 256, 1024), 0.0, 4.6),    # n = 10: an exact optimum fits
    Member(HYPER,         {"degree": 3, "hyper_share": 0.1, "rank": 3}, (16, 64, 256, 1024), 1.0, 0.29),
    Member(HYPER,         {"degree": 3, "hyper_share": 0.1, "rank": 4}, (16, 64, 256, 1024), 1.0, 0.31),
)
REFERENCE_SEEDS = (0, 1, 2)
REFERENCE_BONDS = ("uniform",)

# The open legs: legs family -> shares, one instance per share (one for a family without shares). The legs of
# scattered and region weigh share * W bits, so they keep the same weight relative to the width at every size and in
# every family: 0.5 puts them below the width, 1.5 above. A design choice, not a measurement.
REFERENCE_LEGS = {
    "closed":    (),            # no legs: the network is a number
    "scattered": (0.5,),        # legs on random tensors
    "region":    (0.5, 1.5),    # legs on a connected block, as for a reduced density matrix
    "mps":       (),            # one leg per tensor: a state
}

# The classifier suite: closed networks only (legs change no descriptor), other seeds, and more members and sizes:
# sizes 16-1024 (a lattice at every side L), randreg degrees 3-6, geometric at the reference r_scale and 1.25 x it,
# small-world at four (degree, p), hyper at rank 3 and 4.
CLASSIFIER_SIZES = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
CLASSIFIER_MEMBERS = (
    Member("lattice",    {"dim": 2},        tuple(L**2 for L in range(4, 33))),
    Member("lattice",    {"dim": 3},        tuple(L**3 for L in range(3, 11))),
    Member("randreg",    {"degree": 3},     CLASSIFIER_SIZES),
    Member("randreg",    {"degree": 4},     CLASSIFIER_SIZES),
    Member("randreg",    {"degree": 5},     CLASSIFIER_SIZES),
    Member("randreg",    {"degree": 6},     CLASSIFIER_SIZES),
    Member("geometric",  {"dim": 2, "r_scale": 1.8},  CLASSIFIER_SIZES),
    Member("geometric",  {"dim": 2, "r_scale": 2.25}, CLASSIFIER_SIZES),
    Member("geometric",  {"dim": 3, "r_scale": 1.4},  CLASSIFIER_SIZES),
    Member("geometric",  {"dim": 3, "r_scale": 1.75}, CLASSIFIER_SIZES),
    Member("smallworld", {"degree": 4, "p": 0.05}, CLASSIFIER_SIZES),
    Member("smallworld", {"degree": 4, "p": 0.1},  CLASSIFIER_SIZES),
    Member("smallworld", {"degree": 4, "p": 0.3},  CLASSIFIER_SIZES),
    Member("smallworld", {"degree": 6, "p": 0.1},  CLASSIFIER_SIZES),
    Member("tree",       {},                CLASSIFIER_SIZES),
    Member(HYPER,        {"degree": 3, "hyper_share": 0.1, "rank": 3}, CLASSIFIER_SIZES),
    Member(HYPER,        {"degree": 3, "hyper_share": 0.1, "rank": 4}, CLASSIFIER_SIZES),
)
# ten seeds per (member, size, bond rule); script 02 resumes where it stopped, so the suite can be measured in parts
CLASSIFIER_SEEDS = tuple(range(1000, 1010))
CLASSIFIER_BONDS = ("uniform", "spread")

SUITES = {"reference": Suite(REFERENCE_MEMBERS, REFERENCE_SEEDS, REFERENCE_BONDS, REFERENCE_LEGS),
          "classifier": Suite(CLASSIFIER_MEMBERS, CLASSIFIER_SEEDS, CLASSIFIER_BONDS, {"closed": ()})}


def _legs_k(member: Member, n: int, legs: str, shares: tuple[float, ...]) -> tuple[int, ...]:
    """The number of open legs k of each instance of one (member, size, legs family).

    closed: 0; mps: n; scattered and region: one k per share, k = max(1, round(share * c * n**alpha)), so the legs
    weigh about share * W bits (a leg has dimension chi = 2: one bit). A region has at most n legs: the suite caps k
    (instances.generate would refuse it). ValueError if closed or mps get a share, scattered or region none, or the
    member has no alpha and c."""
    if legs not in LEGS_WITH_K:                  # closed, mps; any other name is refused later by generate
        if shares:
            raise ValueError(f"legs {legs!r} take no share (k is fixed: 0 closed, n mps): got {shares}")
        return (0,) if legs == "closed" else (n,)
    if not shares:
        raise ValueError(f"legs {legs!r} need at least one share (their k is share * W): got none")
    if member.alpha is None or member.c is None:
        raise ValueError(f"{member.family} {member.knobs} has no alpha and c: legs {legs!r} need them "
                         f"(k = share * c * n**alpha)")
    ks = [max(1, round(share * member.c * n ** member.alpha)) for share in shares]
    return tuple(min(k, n) for k in ks) if legs == "region" else tuple(ks)


def plan(suite: Suite, chi: int = CHI) -> list[dict[str, Any]]:
    """The arguments of generate() for every instance of the suite, in order (member, size, seed, bond rule, legs,
    share), duplicates included. Nothing is drawn, so the suite can be counted without being built."""
    wanted = []
    for member in suite.members:
        for n in member.sizes:
            for seed, rule in product(suite.seeds, suite.bonds):    # every (seed, rule) pair, seeds outermost
                for legs, shares in suite.legs.items():
                    for k in _legs_k(member, n, legs, shares):
                        wanted.append(dict(family=member.family, n=n, seed=seed, chi=chi, bonds=rule,
                                           legs=legs, k=k, **member.knobs))
    return wanted


def build(suite: Suite, chi: int = CHI, with_benchmark: bool = False) -> list[Instance]:
    """The instances of plan(), each id once, then the benchmark networks if asked.

    A duplicate is a network the seed does not change (a closed lattice with uniform bonds is the same at every
    seed): the first is kept. len(plan(...)) - len(result) is the number dropped."""
    out, seen = [], set()
    for inst in [generate(**args) for args in plan(suite, chi)]:
        if inst.id not in seen:
            seen.add(inst.id)
            out.append(inst)
    if with_benchmark:
        for name in benchmark.names():
            out.append(benchmark.load(name))
    return out


RECORD_FIELDS = ("family", "params", "seed", "eq", "size_dict", "id")   # one JSON line


def record(inst: Instance) -> dict[str, Any]:
    """The JSON-able dict of an instance: RECORD_FIELDS, structure only, never arrays."""
    return {"family": inst.family, "params": inst.params, "seed": inst.seed,
            "eq": inst.eq, "size_dict": inst.size_dict, "id": inst.id}


def from_record(rec: dict[str, Any]) -> Instance:
    """The inverse of record: the network is rebuilt from the einsum string eq. ValueError if its id is not the stored
    one.

    JSON reads every tuple back as a list. inputs and output are tuples again because they come from eq; the only
    other tuples, the steps of the stored benchmark paths, are converted here. So the rebuilt Instance equals the
    saved one."""
    terms, open_labels = ctg.utils.eq_to_inputs_output(rec["eq"])
    inputs, output = [tuple(t) for t in terms], tuple(open_labels)
    size_dict = rec["size_dict"]
    # the id is a hash of the network: an edited or truncated line fails here instead of giving another network
    if network_id(inputs, output, size_dict) != rec["id"]:
        raise ValueError(f"id mismatch: {rec['id']}")
    params = rec["params"]
    if rec["family"] == "benchmark":
        for stored in params["paths"].values():
            stored["path"] = [tuple(step) for step in stored["path"]]
    return Instance(inputs, output, size_dict, rec["family"], params, rec["seed"])


def save(instances: Iterable[Instance], path: pathlib.Path) -> None:
    """One JSON line per instance (record), UTF-8. The meta file beside it is written by script 01."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # a lone surrogate cannot be written in UTF-8: benchmark.without_surrogates renames those labels on loading
    with open(path, "w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(config.json_line(record(inst)) + "\n")


def load(path: pathlib.Path) -> list[Instance]:
    """The instances of save(), in order, each id checked. config.load_lines(path) gives the records without
    rebuilding the networks."""
    out = []
    for rec in config.load_lines(path):
        try:
            out.append(from_record(rec))
        except ValueError as error:
            raise ValueError(f"{error} in {path}") from None
    return out

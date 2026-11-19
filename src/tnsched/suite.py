"""The dataset: which instances make the suite, and the file they are stored in.

instances.py says how one network is made; this module says which ones are made. A `Suite` is
the menu entry (members, seeds, bond rules, leg recipes) and SUITES names them: "reference" (the
test set) and "classifier" (the training set for the descriptor classifier). Instances are written
as JSON lines: structure only, never arrays, so every line is small and every instance can be
rebuilt and checked by its id.

The module reads top to bottom: Member, Recipe, Suite, the tables (MEMBERS, SEEDS, LEG_RECIPES,
the CLASSIFIER_ ones, SUITES), then the steps of build() in the order it runs them (plan ->
generate each entry -> drop duplicates -> add the benchmark if asked), then the file format
(_record / _from_record, save / load).
"""
from __future__ import annotations

import json
import pathlib
import warnings
from collections.abc import Iterable
from itertools import product              # product(a, b): every pair (x, y), x from a, y from b
from typing import Any, NamedTuple

import cotengra as ctg

from . import benchmark, config
from .instances import CHI, HYPER, Instance, generate, network_id


class Member(NamedTuple):
    """One row of a suite: a family with ALL its knobs, and the sizes it is drawn at."""
    family: str                  # a key of instances.KNOBS
    knobs: dict[str, Any]        # every knob of the family (instances.KNOBS[family]), written out
    sizes: tuple[int, ...]       # exact n: lattices need n = L**D
    alpha: float | None = None   # alpha and c: how the width grows, W = c * n**alpha bits (below); only _legs_args
    c: float | None = None       # reads them, so a suite with no open legs leaves them out


# A NamedTuple is a tuple whose fields also have names: Recipe("mps", None).legs is "mps".
class Recipe(NamedTuple):
    """One row of LEG_RECIPES: which open legs, and how many as a share of the member's width."""
    legs: str                 # a key of instances.LEGS
    share: float | None       # the legs weigh about share * W bits (_legs_args); None where there is
                              # no choice (closed: no legs, mps: one per tensor)


class Suite(NamedTuple):
    """What one file of instances holds: every (member, size, seed, bond rule, recipe) is one instance."""
    members: tuple[Member, ...]
    seeds: tuple[int, ...]
    bonds: tuple[str, ...]        # instances.BOND_RULES lists the ones that exist
    recipes: tuple[Recipe, ...]


# The reference suite: one row per (family, knobs) pair, and the sizes it is drawn at. Adding a row,
# or a size, is all it takes to grow the dataset. It is the test set: its seeds are not the classifier's.
#   alpha, c   how the width of the network grows: W = c * n**alpha bits, W the width of cotengra's
#              greedy path on the closed network, measured at seeds 0-2 at the sizes below. alpha is fixed by
#              theory (tree 0; 2D lattice and geometric graph 1/2; 3D 2/3; random regular,
#              small-world and hyperedge networks 1); c is the geometric mean of W / n**alpha,
#              2 digits. Only _legs_args reads them, to say how many open legs weigh a given share of W.
#              They are frozen numbers: a new c changes the stored file, and the slow test of test_suite says so.
#   How good the fit is: loose where the theory's alpha meets a small n. The worst ratio of the
#   fit to the measured W over the sizes and seeds below: smallworld x2.6, geometric dim 3 x2.0,
#   geometric dim 2 x1.8, tree x1.5 (its W grows from 3 to 7 over n = 10..1024 while alpha = 0 says
#   constant), the others x1.1-1.4. So the k of a recipe is a design choice (legs that weigh about a
#   share of W, in the same proportion at every size), not a measurement of W: a path finder
#   measures it.
#   r_scale of "geometric": chosen for few redraws (instances.py, FAMILIES): 1.8 at dim 2, 1.4 at dim 3.
MEMBERS = (
    Member("lattice",     {"dim": 2},        (16, 64, 256, 1024),     0.5, 1.5),    # L = 4, 8, 16, 32
    Member("lattice",     {"dim": 3},        (27, 64, 216, 1000),     2 / 3, 1.7),  # L = 3, 4, 6, 10
    Member("randreg",     {"degree": 3},     (16, 64, 256, 1024),     1.0, 0.23),
    Member("randreg",     {"degree": 4},     (16, 64, 256, 1024),     1.0, 0.45),
    Member("geometric",   {"dim": 2, "r_scale": 1.8}, (16, 64, 256, 1024), 0.5, 3.5),
    Member("geometric",   {"dim": 3, "r_scale": 1.4}, (16, 64, 256, 1024), 2 / 3, 2.0),
    Member("smallworld",  {"degree": 4, "p": 0.1}, (16, 64, 256, 1024), 1.0, 0.19),
    Member("tree",        {},                (10, 16, 64, 256, 1024), 0.0, 4.6),    # n = 10: small enough for an exact optimum
    Member(HYPER,         {"degree": 3, "hyper_share": 0.1, "rank": 3}, (16, 64, 256, 1024), 1.0, 0.29),
    Member(HYPER,         {"degree": 3, "hyper_share": 0.1, "rank": 4}, (16, 64, 256, 1024), 1.0, 0.31),
)
SEEDS = (0, 1, 2)           # the stored seeds of every member and size
BONDS = ("uniform",)        # the stored bond rules

# What the reference suite holds per (member, size, seed, bond rule): one instance per row. The legs of
# scattered and region weigh share * W bits, W = c * n**alpha of the member, so they keep the same
# weight relative to the network's width at every size and in every family. Whether they dominate
# the cost is a statement about cost, measured with a path finder, not decided here.
LEG_RECIPES = (
    Recipe("closed",    None),     # the closed network: a number
    Recipe("scattered", 0.5),      # legs on random tensors
    Recipe("region",    0.5),      # legs on a connected block, as for a reduced density matrix
    Recipe("region",    1.5),      # the same block, legs 1.5 times the width
    Recipe("mps",       None),     # one leg per tensor: a state
)

# The classifier suite: closed networks only (the legs change no descriptor), many more members, sizes
# and seeds than the reference, and other seeds. Sizes spread from 16 to 1024 (a lattice at every side
# L); randreg degrees 3-6; geometric at the reference r_scale and 1.25 times it; small-world at four
# (degree, p); hyper at rank 3 and 4.
SPREAD = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
CLASSIFIER_MEMBERS = (
    Member("lattice",    {"dim": 2},        tuple(L**2 for L in range(4, 33))),
    Member("lattice",    {"dim": 3},        tuple(L**3 for L in range(3, 11))),
    Member("randreg",    {"degree": 3},     SPREAD),
    Member("randreg",    {"degree": 4},     SPREAD),
    Member("randreg",    {"degree": 5},     SPREAD),
    Member("randreg",    {"degree": 6},     SPREAD),
    Member("geometric",  {"dim": 2, "r_scale": 1.8},  SPREAD),
    Member("geometric",  {"dim": 2, "r_scale": 2.25}, SPREAD),
    Member("geometric",  {"dim": 3, "r_scale": 1.4},  SPREAD),
    Member("geometric",  {"dim": 3, "r_scale": 1.75}, SPREAD),
    Member("smallworld", {"degree": 4, "p": 0.05}, SPREAD),
    Member("smallworld", {"degree": 4, "p": 0.1},  SPREAD),
    Member("smallworld", {"degree": 4, "p": 0.3},  SPREAD),
    Member("smallworld", {"degree": 6, "p": 0.1},  SPREAD),
    Member("tree",       {},                SPREAD),
    Member(HYPER,        {"degree": 3, "hyper_share": 0.1, "rank": 3}, SPREAD),
    Member(HYPER,        {"degree": 3, "hyper_share": 0.1, "rank": 4}, SPREAD),
)
CLASSIFIER_SEEDS = tuple(range(1000, 1010))     # ten seeds per (member, size, bond rule): 02 resumes where it stopped, so the suite can be measured in several runs
CLASSIFIER_BONDS = ("uniform", "spread")

SUITES = {"reference": Suite(MEMBERS, SEEDS, BONDS, LEG_RECIPES),
          "classifier": Suite(CLASSIFIER_MEMBERS, CLASSIFIER_SEEDS, CLASSIFIER_BONDS, (Recipe("closed", None),))}


def _legs_args(member: Member, n: int, recipe: Recipe) -> dict[str, Any]:
    """The legs and k of one (member, size, recipe): closed k = 0, mps k = n, else the legs that weigh about
    share * W bits (d_open = chi = 2: one bit each): k = max(1, round(share * c * n**alpha)), at most n for a region.
    A member without alpha and c gets a warning and alpha = c = 1: the legs weigh share * n bits. Pure: nothing is drawn."""
    if recipe.legs == "closed":
        return dict(legs="closed", k=0)
    if recipe.legs == "mps":
        return dict(legs="mps", k=n)
    alpha, c = member.alpha, member.c
    if alpha is None or c is None:
        warnings.warn(f"{member.family} {member.knobs} has no alpha and c: its legs weigh share * n bits (alpha = c = 1)")
        alpha, c = 1.0, 1.0
    k = max(1, round(recipe.share * c * n ** alpha))
    return dict(legs=recipe.legs, k=min(k, n) if recipe.legs == "region" else k)


def plan(suite: Suite, chi: int = CHI) -> list[dict[str, Any]]:
    """What the suite asks of generate(): one keyword dict per instance, in the order of the suite
    (member, size, seed, bond rule, then its recipes), the member's knobs written out
    and k always given. Pure: nothing is drawn, so a test or a script can count or inspect the
    suite without building it. Duplicates (see build) are still in."""
    wanted = []
    for member in suite.members:
        for n in member.sizes:
            for seed, rule in product(suite.seeds, suite.bonds):    # every (seed, rule) pair, seeds outermost
                for recipe in suite.recipes:
                    wanted.append(dict(family=member.family, n=n, seed=seed, chi=chi, bonds=rule,
                                       **_legs_args(member, n, recipe), **member.knobs))
    return wanted


def build(suite: Suite, chi: int = CHI, with_benchmark: bool = False) -> list[Instance]:
    """[generate(**args) for args in plan(...)], duplicates dropped, the benchmark added if asked.

    Duplicates (same id, so the same network) are kept once, the first. They arise where
    the seed changes nothing: a closed lattice with uniform bonds is the same network for every
    seed, so the three seeds give one instance. How many were dropped is len(plan(...)) minus
    len(result), worked out where it is reported. Every error of generate() passes through."""
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


def _record(inst: Instance) -> dict[str, Any]:
    """The JSON-able dict of an instance: RECORD_FIELDS, structure only, never arrays."""
    return {"family": inst.family, "params": inst.params, "seed": inst.seed,
            "eq": inst.eq, "size_dict": inst.size_dict, "id": inst.id}


def _from_record(rec: dict[str, Any]) -> Instance:
    """The inverse of _record. ValueError if the network rebuilt from the einsum string does not
    have the stored id.

    JSON changes types: a tuple is written as a list and read back as a list. The network is
    rebuilt from the string, so inputs and output are tuples again; the only tuples left in params
    are the steps of the stored benchmark paths (params["paths"][key]["path"]), turned back here.
    So Instance equality survives the round trip, for the benchmark too."""
    terms, open_labels = ctg.utils.eq_to_inputs_output(rec["eq"])
    inputs, output = [tuple(t) for t in terms], tuple(open_labels)
    size_dict = rec["size_dict"]
    # the id is the network's own hash: a hand-edited or truncated line fails here
    # instead of giving a quietly different network
    if network_id(inputs, output, size_dict) != rec["id"]:
        raise ValueError(f"id mismatch: {rec['id']}")
    params = rec["params"]
    if rec["family"] == "benchmark":
        for stored in params["paths"].values():
            stored["path"] = [tuple(step) for step in stored["path"]]
    return Instance(inputs, output, size_dict, rec["family"], params, rec["seed"])


def save(instances: Iterable[Instance], path: pathlib.Path) -> None:
    """One JSON line per instance (_record). The meta file next to it is written by the script that builds the
    suite (config.write_meta)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # json_line writes a non-ASCII label (the benchmark has some) as the character itself; the file is UTF-8. A lone
    # surrogate could not be written this way at all, which is why benchmark.without_surrogates renames them on loading.
    with open(path, "w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(config.json_line(_record(inst)) + "\n")


def load(path: pathlib.Path) -> list[Instance]:
    """The instances of save(), in order; each id is checked against the stored one."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(_from_record(json.loads(line)))
            except ValueError as error:
                raise ValueError(f"{error} in {path}") from None
    return out

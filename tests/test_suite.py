"""suite.py: the tables (REFERENCE_MEMBERS, the legs and their shares, the two suites), the k of the legs, plan, build,
and the JSON file of a suite. The tables first, then plan (nothing is built), then build, save and load; the slow
tests build the whole reference suite."""
import dataclasses
from collections import Counter

import cotengra as ctg
import networkx as nx
import pytest
from conftest import gen

from tnsched import config, instances, suite
from tnsched.instances import HYPER
from tnsched.suite import REFERENCE_LEGS, REFERENCE_MEMBERS, Member, Suite

REFERENCE, CLASSIFIER = suite.SUITES["reference"], suite.SUITES["classifier"]


def sizes_of(family, **knobs):
    """The sizes of the one member with this family and these knobs."""
    return next(m.sizes for m in REFERENCE_MEMBERS if m.family == family and m.knobs == knobs)


# --- the tables ---------------------------------------------------------------------------------

def k_of(member, n, share, legs="scattered"):
    """The k that plan() gives one (member, size, legs, share)."""
    [args] = suite.plan(Suite((member._replace(sizes=(n,)),), (0,), ("uniform",), {legs: (share,)}))
    return args["k"]


def test_the_k_of_the_legs_is_round_share_c_n_to_the_alpha_at_least_one():
    """k = max(1, round(share * c * n**alpha)), at most n for a region."""
    lattice = next(m for m in REFERENCE_MEMBERS if m.family == "lattice" and m.knobs == {"dim": 2})
    assert k_of(lattice, 64, 0.5) == round(0.5 * 1.5 * 64 ** 0.5) == 6
    assert k_of(lattice, 1024, 1.5, "region") == 72
    assert k_of(next(m for m in REFERENCE_MEMBERS if m.family == "tree"), 1024, 0.5) == 2                  # alpha 0: the same at every size
    assert k_of(Member("tree", {}, (4,), 0.0, 0.1), 4, 0.5) == 1                                   # never fewer than one leg
    wide = Member("tree", {}, (4,), 1.0, 10.0)
    assert k_of(wide, 4, 1.5, "region") == 4                                                       # a region holds at most n tensors
    assert k_of(wide, 4, 1.5, "scattered") == 60                                                   # scattered draws with replacement


@pytest.mark.parametrize("legs", ["scattered", "region"])
def test_a_member_without_alpha_and_c_cannot_get_legs_with_k(legs):
    with pytest.raises(ValueError, match="no alpha and c"):
        k_of(CLASSIFIER.members[0], 16, 0.5, legs)
    assert suite.plan(CLASSIFIER._replace(members=CLASSIFIER.members[:1]))                       # closed needs neither


@pytest.mark.parametrize("legs, shares", [("scattered", ()), ("region", ()), ("closed", (0.5,)), ("mps", (1.0,))])
def test_scattered_and_region_need_a_share_and_closed_and_mps_take_none(legs, shares):
    lattice = REFERENCE_MEMBERS[0]._replace(sizes=(16,))
    with pytest.raises(ValueError, match=legs):
        suite.plan(Suite((lattice,), (0,), ("uniform",), {legs: shares}))


def test_the_members_cover_every_family_with_every_knob_once():
    """A family added to KNOBS cannot be forgotten in the suite; a knob cannot be misspelt or left out."""
    assert {m.family for m in REFERENCE_MEMBERS} == set(instances.KNOBS)
    for m in REFERENCE_MEMBERS:
        assert set(m.knobs) == set(instances.KNOBS[m.family]), m
        assert m.sizes == tuple(sorted(m.sizes)) and m.c > 0 and 0 <= m.alpha <= 1, m
    assert len({(m.family, tuple(sorted(m.knobs.items()))) for m in REFERENCE_MEMBERS}) == len(REFERENCE_MEMBERS)   # one row per pair


def test_the_classifier_suite_is_closed_with_other_seeds_and_both_bond_rules():
    assert CLASSIFIER.legs == {"closed": ()} and CLASSIFIER.bonds == instances.BOND_RULES
    assert not set(CLASSIFIER.seeds) & set(REFERENCE.seeds) and CLASSIFIER.seeds == tuple(range(1000, 1010))
    assert {m.family for m in CLASSIFIER.members} == set(instances.KNOBS)
    for m in CLASSIFIER.members:
        assert set(m.knobs) == set(instances.KNOBS[m.family]) and m.sizes == tuple(sorted(m.sizes)), m
        assert m.alpha is None and m.c is None                                                    # nothing here reads them: no legs
    assert len({(m.family, tuple(sorted(m.knobs.items()))) for m in CLASSIFIER.members}) == len(CLASSIFIER.members)


# --- plan: what would be drawn, without drawing it -------------------------------------------------------

@pytest.mark.parametrize("name", list(suite.SUITES))
def test_plan_is_pure_and_every_entry_is_a_valid_call(name):
    chosen = suite.SUITES[name]
    wanted = suite.plan(chosen)
    assert suite.plan(chosen) == wanted                                                    # nothing is drawn, nothing changes
    per_size = sum(max(1, len(shares)) for shares in chosen.legs.values())                 # one instance per share, one without
    assert len(wanted) == sum(len(m.sizes) for m in chosen.members) * len(chosen.seeds) * len(chosen.bonds) * per_size
    for a in wanted:
        knobs = {k: a[k] for k in instances.KNOBS[a["family"]]}
        instances._check_args(a["family"], a["n"], a["seed"], a["chi"], a["bonds"], None, a["legs"], a["k"], knobs)
    if name == "reference":
        assert {a["rank"] for a in wanted if a["family"] == HYPER} == {3, 4}


def test_plan_puts_every_legs_share_at_every_size_of_every_member_with_k_from_its_share():
    """One entry per (member, size, legs, share) in the order of REFERENCE_LEGS; k from its share; mps has k = n, closed has k = 0."""
    rows = [(legs, share) for legs, shares in REFERENCE_LEGS.items() for share in (shares or (None,))]   # None: no share
    assert [legs for legs, _ in rows] == ["closed", "scattered", "region", "region", "mps"]
    for member in REFERENCE_MEMBERS:
        entries = suite.plan(REFERENCE._replace(members=(member,), seeds=(0,)))
        assert len(entries) == len(member.sizes) * len(rows)
        for args, (legs, share) in zip(entries, rows * len(member.sizes)):
            assert args["legs"] == legs and args["n"] in member.sizes
            if share is not None:
                k = max(1, round(share * member.c * args["n"] ** member.alpha))
                assert args["k"] == (min(k, args["n"]) if legs == "region" else k)
            assert args["k"] == {"mps": args["n"], "closed": 0}.get(args["legs"], args["k"])
    largest = [a for a in suite.plan(REFERENCE) if a["legs"] == "mps" and a["n"] >= 1000]          # no size limit: 2**1024 entries
    assert len(largest) >= 10 and all(a["k"] == a["n"] for a in largest)


# --- build, save and load -------------------------------------------------------------------------

def test_build_keeps_one_instance_per_id_in_plan_order():
    """A closed uniform lattice is one network for every seed: the three seeds give one instance; a tree gives three."""
    lattice = next(m for m in REFERENCE_MEMBERS if m.family == "lattice" and m.knobs == {"dim": 2})._replace(sizes=(16,))
    tree = next(m for m in REFERENCE_MEMBERS if m.family == "tree")._replace(sizes=(16,))
    built = suite.build(REFERENCE._replace(members=(lattice, tree), legs={"closed": ()}, seeds=(0, 1, 2)))
    assert [i.family for i in built] == ["lattice"] + ["tree"] * 3 and len({i.id for i in built}) == 4


def test_save_then_load_gives_the_same_instances_and_a_network_of_line_terminators_survives(tmp_path):
    """Every field comes back, provenance included; U+0085, U+2028 and U+2029 are written escaped: one line per instance."""
    insts = [gen("lattice", 27, 0, dim=3), gen(HYPER, 16, 1), gen("tree", 8, 2, bonds="spread"), gen("smallworld", 32, 0, p=0.5),
             gen("lattice", 16, 0, legs="region", k=4), gen("randreg", 16, 0, legs="scattered", k=5), gen("tree", 8, 0, legs="mps")]
    suite.save(insts, tmp_path / "suite.jsonl")
    back = suite.load(tmp_path / "suite.jsonl")
    assert back == insts and [b.id for b in back] == [a.id for a in insts] and [b.name for b in back] == [a.name for a in insts]
    assert nx.utils.graphs_equal(back[1].graph, insts[1].graph)
    inst = gen("tree", 8, 0)
    odd = {inst.inputs[0][0]: "\u0085", inst.inputs[1][0]: " ", inst.inputs[2][1]: " "}
    rename = lambda term: [odd.get(x, x) for x in term]
    weird = dataclasses.replace(inst, inputs=[rename(t) for t in inst.inputs], output=rename(inst.output),
                                size_dict={odd.get(k, k): v for k, v in inst.size_dict.items()})
    suite.save([weird, inst], tmp_path / "odd.jsonl")
    assert len((tmp_path / "odd.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert suite.load(tmp_path / "odd.jsonl")[0].id == weird.id != inst.id


def test_records_are_the_json_lines_without_building_any_network(tmp_path):
    insts = [gen("lattice", 16, 0), gen(HYPER, 16, 1, legs="scattered", k=3)]
    suite.save(insts, tmp_path / "suite.jsonl")
    recs = config.load_lines(tmp_path / "suite.jsonl")
    assert recs == [suite.record(i) for i in insts] and [r["id"] for r in recs] == [i.id for i in insts]
    assert set(recs[0]) == set(suite.RECORD_FIELDS) and suite.from_record(recs[1]) == insts[1]


def test_load_refuses_a_record_whose_id_does_not_match_and_names_the_file(tmp_path):
    """load rebuilds every network from its einsum string and checks the id it stored: an edited line fails here."""
    inst = gen("lattice", 16, 0)
    path = tmp_path / "suite.jsonl"
    suite.save([inst], path)
    path.write_text(path.read_text(encoding="utf-8").replace(inst.id, "0" * 16), encoding="utf-8")
    with pytest.raises(ValueError, match=r"id mismatch.*suite\.jsonl"):
        suite.load(path)


def test_build_lets_every_error_of_generate_through(monkeypatch):
    def broken(**args):
        raise ValueError("a bug")
    monkeypatch.setattr(suite, "generate", broken)
    with pytest.raises(ValueError, match="a bug"):
        suite.build(REFERENCE._replace(members=(REFERENCE_MEMBERS[0],), seeds=(0,)))
    monkeypatch.undo()
    def far_too_sparse(n, s, dim, r_scale):
        return nx.random_geometric_graph(n, 0.001, dim=2, seed=s)                                  # r so small: no edge at all
    monkeypatch.setitem(instances.FAMILIES, "geometric", far_too_sparse)
    member = Member("geometric", {"dim": 2, "r_scale": 1.8}, (16,), 0.5, 3.5)
    with pytest.raises(ValueError, match="no connected draw"):                                     # draw_connected gives up: a bug in a knob
        suite.build(REFERENCE._replace(members=(member,), seeds=(0,)))


# --- the whole reference suite (slow) ------------------------------------------------------------

@pytest.mark.slow
def test_the_reference_suite_has_583_instances_with_unique_ids_and_names():
    built = suite.build(REFERENCE)
    ids, names = [i.id for i in built], [i.name for i in built]
    assert len(built) == 583 and len(set(ids)) == len(ids) and len(set(names)) == len(names)
    closed = Counter((i.family, i.params["knobs"].get("dim")) for i in built if i.params["legs"] == "closed")
    assert closed["lattice", 2] == len(sizes_of("lattice", dim=2))                                  # a uniform lattice ignores the seed
    closed_names = Counter(i.name.split("-n")[0] for i in built if i.params["legs"] == "closed")
    assert closed_names["randreg"] == 3 * len(sizes_of("randreg", degree=3)) + 3 * len(sizes_of("randreg", degree=4))
    assert all(i.params["n"] in next(m.sizes for m in REFERENCE_MEMBERS if m.family == i.family
                                     and all(i.params["knobs"][k] == v for k, v in m.knobs.items())) for i in built)
    assert all(i.params["log2_total"] >= i.params["log2_output"] for i in built)


@pytest.mark.slow
def test_every_instance_of_the_reference_suite_is_connected(reference):
    for inst in reference:
        assert inst.components == 1 and inst.params["attempt"] < 10, inst.name
        assert inst.params["connected_by"] == instances.CONNECTED_BY[inst.family], inst.name
        if inst.params["connected_by"] == "construction":
            assert inst.params["attempt"] == 0, inst.name


@pytest.mark.slow
def test_the_frozen_calibration_still_describes_the_generators():
    """alpha and c of REFERENCE_MEMBERS were fitted to the greedy width W of the closed instances: W stays within a factor 3 of
    c * n**alpha at every size and seed. If a generator changes, this fails: refit."""
    for m in REFERENCE_MEMBERS:
        for n in m.sizes:
            for seed in suite.REFERENCE_SEEDS:
                inst = gen(m.family, n, seed, **m.knobs)
                w = ctg.array_contract_tree(*inst.network, optimize="greedy").contraction_width()
                assert 1 / 3 < w / (m.c * n ** m.alpha) < 3, (m.family, m.knobs, n, seed, w)


@pytest.mark.slow
def test_the_stored_reference_file_is_what_build_gives():
    """results/instances/reference.jsonl against a fresh build(): the same ids in the same order. Other library
    versions may draw other networks, so a mismatch prints the versions of both."""
    path = config.INSTANCES_DIR / "reference.jsonl"
    if not path.exists():
        pytest.skip("no results/instances/reference.jsonl: run scripts/01_build_instances.py")
    stored = [i.id for i in suite.load(path) if i.family != "benchmark"]
    fresh = [i.id for i in suite.build(REFERENCE)]
    if stored != fresh:
        was = config.meta_path(path).read_text(encoding="utf-8") if config.meta_path(path).exists() else "no meta file"
        pytest.fail(f"reference.jsonl differs from build(): {len(stored)} stored, {len(fresh)} built.\n"
                    f"file built with: {was}\nthis run: {config.library_versions()}")

"""Shared by the tests: gen(), signature(), component(), graph_component(), TEST_TARGETS and files_of(), and the two
suite fixtures."""
import networkx as nx
import pytest

from tnsched import descriptors, instances, suite
from tnsched.features import component_input


def gen(family: str, n: int, *args, **kwargs) -> instances.Instance:
    """instances.generate(family, n, *args, **kwargs), where a knob of the family that the call leaves out is
    taken from the first row of suite.REFERENCE_MEMBERS of that family. generate() itself needs every knob; the tests
    are not a second table of numbers."""
    first = next((m for m in suite.REFERENCE_MEMBERS if m.family == family), None)       # None: an unknown family, for generate to refuse
    knobs = {} if first is None else {name: value for name, value in first.knobs.items() if name not in kwargs}
    return instances.generate(family, n, *args, **kwargs, **knobs)


def signature(inputs, output, size_dict):
    """The network up to the names of its labels and the order of axes in a tensor: for each label the tensors that
    carry it (with multiplicity), its dimension and its position in the output (-1 if it is not open), sorted."""
    carriers = {e: [] for e in size_dict}
    for v, term in enumerate(inputs):
        for e in term:
            carriers[e].append(v)
    return sorted((tuple(sorted(c)), size_dict[e], output.index(e) if e in output else -1) for e, c in carriers.items())


def component(inputs, output=(), sizes=None) -> descriptors.ConnectedComponent:
    """The largest connected component of the core of a hand-made network; every label has dimension 2 unless `sizes` says."""
    labels = {label for term in inputs for label in term} | set(output)
    inst = instances.Instance([tuple(t) for t in inputs], tuple(output), sizes or dict.fromkeys(labels, 2), "benchmark",
                              {"n": len(inputs), "name": "hand"}, None)
    return descriptors.split(inst.core, 0)[0]


def graph_component(G: nx.Graph) -> descriptors.ConnectedComponent:
    """The component whose H is the graph G of bonds (weight 1 unless an edge says), its vertices renumbered 0..n-1."""
    return descriptors.ConnectedComponent(nx.convert_node_labels_to_integers(G, ordering="sorted"), 0)


TEST_TARGETS = ("tw_lower", "planar", "forman_mean", "ball_dimension")     # one of every kind: log, binary, linear, a fit


def files_of(insts):
    """What scripts 01, 02 and 03 write for `insts` (the fast descriptors only): the records of the instances file, the
    components of each instance (instance id -> [(component id, tensors)]), and the row and the input of each distinct
    component, by id."""
    components, rows, inputs = {}, {}, {}
    for inst in insts:
        pieces = descriptors.split(inst.core, 0)
        components[inst.id] = [(c.id, c.n) for c in pieces]
        for c in pieces:
            if c.id not in rows:
                rows[c.id], inputs[c.id] = descriptors.measure(c, slow=False), component_input(c)
    return [suite.record(inst) for inst in insts], components, rows, inputs


@pytest.fixture(scope="session")
def small_suite() -> list[instances.Instance]:
    """Every member of the reference suite at its two smallest sizes, seed 0, every legs family and share, both bond rules."""
    reference = suite.SUITES["reference"]
    members = tuple(m._replace(sizes=m.sizes[:2]) for m in reference.members)
    return suite.build(reference._replace(members=members, seeds=(0,), bonds=instances.BOND_RULES))


@pytest.fixture(scope="session")
def reference() -> list[instances.Instance]:
    """The whole reference suite with both bond rules, built once (the slow tests share it)."""
    return suite.build(suite.SUITES["reference"]._replace(bonds=instances.BOND_RULES))

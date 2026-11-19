"""config.py (a module this small is tested here), scripts/_shared.py and the scripts 00 to 03: the JSON form of every file and
its meta, which instances file a script works on, the components left to compute, the download of the benchmark, and the
chain 01 -> 02 -> 03. Under pytest there is no keyboard, so the questions are not asked: the flags say everything."""
import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import pickle
import runpy
import sys
import time
import zipfile
from contextlib import nullcontext

import numpy as np
import pytest
from conftest import gen

from tnsched import benchmark, config, descriptors, features, suite

sys.path.insert(0, str(config.ROOT / "scripts"))
import _shared                                                  # noqa: E402


@pytest.fixture(autouse=True)
def results_in_a_temporary_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INSTANCES", tmp_path / "instances")
    monkeypatch.setattr(config, "DESCRIPTORS", tmp_path / "descriptors")
    monkeypatch.setattr(config, "FEATURES", tmp_path / "features")


def parse(*flags):
    ap = argparse.ArgumentParser()
    _shared.add_arguments(ap)
    return ap.parse_args(flags)


def script(name):
    """scripts/<name>.py as a module (their names start with a digit, so they cannot be imported by name)."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name, *flags, monkeypatch):
    """main() of a script, with these flags on its command line."""
    monkeypatch.setattr(sys, "argv", [name, *flags])
    script(name).main()


SMALL = ("--seeds", "1000", "--max-n", "16")
LATTICE = ("--family", "lattice", "--knobs", '{"dim": 2}', "--sizes", "16", "25", "--seeds", "0")


# --- config: the JSON form and the meta file -------------------------------------------------------------

def test_json_line_is_one_line_even_for_splitlines_and_json_loads_gives_the_value_back():
    value = {"label": "a\u0085b c d", "list": [1, 2.5, None], "name": "café"}
    line = config.json_line(value)
    assert len(line.splitlines()) == 1 and "\\u2028" in line and "\\u0085" in line and "\\u2029" in line
    assert "café" in line and ", " not in line and ": " not in line                           # UTF-8 as it is, compact
    assert json.loads(line) == value


def test_write_meta_adds_the_library_versions_and_the_date_and_writes_paths_as_text(tmp_path):
    path = tmp_path / "d" / "x.jsonl"
    config.write_meta(path, request_id="abc", flags={"suite": tmp_path})
    meta = config.read_meta(path, request_id="abc")
    assert meta["flags"] == {"suite": str(tmp_path)} and {"python", "numpy", "scipy", "networkx", "cotengra", "date"} <= set(meta)
    assert "torch" not in meta and config.meta_path(path).name == "x.meta.json"
    assert len(config.meta_path(path).read_text(encoding="utf-8").splitlines()) == 1                # a meta is one line


def test_read_meta_names_the_file_and_every_key_that_differs_and_ends_with_the_advice(tmp_path):
    path = tmp_path / "x.jsonl"
    config.write_meta(path, a=1, b=2)
    with pytest.raises(ValueError, match=r"x\.jsonl.*a: the file has 1, now 5.*b: the file has 2, now 6.*run it again$"):
        config.read_meta(path, "run it again", a=5, b=6)
    with pytest.raises(ValueError, match="no meta file.*run it again"):
        config.read_meta(tmp_path / "y.jsonl", "run it again")


def test_the_paths_of_the_files_are_named_after_the_instances_file():
    path = config.INSTANCES / "small_classifier.jsonl"
    assert config.descriptors_path(path) == config.DESCRIPTORS / "small_classifier.jsonl"
    assert config.features_path(path) == config.FEATURES / "small_classifier.npz"


def test_the_library_versions_leave_out_what_is_not_installed(monkeypatch):
    def missing(name):
        raise config.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(config.metadata, "version", missing)
    assert set(config.library_versions()) == {"python"}


# --- _shared: which instances file, and what is left to compute ------------------------------------------------

def test_build_instances_writes_the_file_and_a_meta_with_the_request_its_id_and_the_flags():
    path = _shared.build_instances("classifier", parse(*SMALL))
    assert path.parent == config.INSTANCES and path.name.startswith("classifier_20")              # --seeds changed the suite: a time
    meta = config.read_meta(path)
    assert meta["seeds"] == [1000] and meta["bonds"] == ["uniform", "spread"] and meta["chi"] == 2 and meta["benchmark"] is False
    assert meta["n_instances"] == len(suite.load(path)) and len(meta["request_id"]) == 16
    assert meta["flags"]["max_n"] == 16 and meta["members"][0] == {"family": "lattice", "knobs": {"dim": 2}, "sizes": [16], "alpha": None, "c": None}
    assert max(i.params["n"] for i in suite.load(path)) == 16


def test_a_tag_puts_a_new_file_next_to_the_one_that_is_there(capsys):
    changed = _shared.build_instances("classifier", parse(*SMALL))
    assert changed.name.startswith("classifier_20") and changed.suffix == ".jsonl"                  # a changed suite has the time in its name
    assert _shared.build_instances("classifier", parse(*SMALL, "--tag", "small")).name == "small_classifier.jsonl"
    manual = _shared.build_instances("manual", parse(*LATTICE))
    assert manual.name.startswith("manual_20") and len(suite.load(manual)) == 2
    assert _shared.build_instances("manual", parse(*LATTICE, "--tag", "t")).name == "t_manual.jsonl"
    capsys.readouterr()
    assert _shared.build_instances("manual", parse(*LATTICE, "--tag", "t")).name == "t_manual.jsonl"   # the same request again: the file is used
    assert "made for this request" in capsys.readouterr().out


def test_two_manual_builds_give_two_files():
    first = _shared.build_instances("manual", parse(*LATTICE))
    time.sleep(1.1)                                                                                  # the name has the time to the second
    other = _shared.build_instances("manual", parse("--family", "tree", "--knobs", "{}", "--sizes", "16", "--seeds", "0"))
    assert other != first and {i.family for i in suite.load(other)} == {"tree"} and {i.family for i in suite.load(first)} == {"lattice"}


def test_a_benchmark_name_ends_with_benchmark(monkeypatch):
    monkeypatch.setattr(suite, "build", lambda chosen, chi, with_benchmark: [])
    assert _shared.build_instances("reference", parse("--benchmark", "--tag", "b")).name == "b_reference_benchmark.jsonl"


def test_manual_is_one_closed_family_from_the_flags_and_needs_them_all():
    with pytest.raises(SystemExit, match="--sizes"):
        _shared.build_instances("manual", parse("--family", "lattice", "--knobs", '{"dim": 2}', "--seeds", "0"))
    with pytest.raises(SystemExit, match=r"--knobs must be a JSON object.*--knobs '\{\"dim\": 2\}'.*PowerShell.*got 'dim=2'"):
        _shared.build_instances("manual", parse("--family", "lattice", "--knobs", "dim=2", "--sizes", "16", "--seeds", "0"))


def test_a_file_built_with_other_flags_is_not_reused_and_a_file_without_its_meta_is_not_used():
    _shared.build_instances("classifier", parse(*SMALL, "--tag", "small"))
    with pytest.raises(SystemExit, match=r"small_classifier\.jsonl.*seeds: the file has \[1000\], now \[1001\].*--tag"):
        _shared.build_instances("classifier", parse("--seeds", "1001", "--max-n", "16", "--tag", "small"))
    path = _shared.build_instances("reference", parse("--max-n", "16", "--tag", "x"))
    config.meta_path(path).unlink()
    with pytest.raises(SystemExit, match="no meta file"):
        _shared.build_instances("reference", parse("--max-n", "16", "--tag", "x"))


def test_pick_instances_takes_a_file_a_build_or_says_what_to_pass():
    built = _shared.build_instances("classifier", parse(*SMALL, "--tag", "s"))
    assert _shared.pick_instances(parse("--suite", str(built))) == built
    assert _shared.pick_instances(parse("--build", "classifier", *SMALL, "--tag", "s")) == built
    with pytest.raises(SystemExit, match="01_build_instances"):
        _shared.pick_instances(parse("--suite", str(config.INSTANCES / "nope.jsonl")))
    with pytest.raises(SystemExit, match="--suite PATH or --build"):
        _shared.pick_instances(parse())


def test_new_components_gives_each_distinct_component_once_skips_the_done_ones_and_keeps_no_graph():
    insts = [gen("randreg", 16, 0), gen("randreg", 16, 0, legs="scattered", k=3), gen("tree", 16, 0)]
    ids = [c.id for c in _shared.new_components(insts, 0, set())]
    assert len(ids) == 2 and len(set(ids)) == 2                                                    # the two randreg are one core
    assert [c.id for c in _shared.new_components(insts, 0, {ids[0]})] == [ids[1]]
    assert all("core" not in i.__dict__ and "graph" not in i.__dict__ for i in insts)              # forget_graphs
    assert list(_shared.new_components(insts, 0, set(ids))) == []


def test_new_components_hands_them_out_one_instance_at_a_time():
    insts = [gen("randreg", 16, 0), gen("tree", 16, 0)]
    pieces = _shared.new_components(insts, 0, set())
    next(pieces)
    assert "core" not in insts[0].__dict__ and "core" not in insts[1].__dict__                      # the first is split and let go, the second has not had its turn
    assert [c.id for c in itertools.islice(pieces, 1)] == [descriptors.split(insts[1].core, 0)[0].id]


def test_with_a_keyboard_the_shared_asks_and_takes_the_answer(monkeypatch):
    _shared.build_instances("classifier", parse(*SMALL, "--tag", "k"))
    other = parse("--seeds", "1001", "--max-n", "16", "--tag", "k")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda question: "second")
    assert _shared.build_instances("classifier", other).name == "second_k_classifier.jsonl"           # a tag for another file
    monkeypatch.setattr("builtins.input", lambda question: "")
    with pytest.raises(SystemExit, match="--tag"):                                                    # Enter stops
        _shared.build_instances("classifier", parse("--seeds", "1002", "--max-n", "16", "--tag", "k"))

    def nobody(question):
        raise EOFError
    monkeypatch.setattr("builtins.input", nobody)
    with pytest.raises(SystemExit, match="--tag"):                                                    # no keyboard at all stops too
        _shared.build_instances("classifier", parse("--seeds", "1003", "--max-n", "16", "--tag", "k"))
    files = sorted(config.INSTANCES.glob("*.jsonl"))
    monkeypatch.setattr("builtins.input", lambda question: "1")
    assert _shared.pick_instances(parse()) == files[0]                                                  # the menu: a file by its number
    monkeypatch.setattr("builtins.input", lambda question: "none of these")
    with pytest.raises(SystemExit, match="neither a file number nor a suite"):
        _shared.pick_instances(parse())
    monkeypatch.setattr("builtins.input", lambda question: "classifier")
    assert _shared.pick_instances(parse(*SMALL, "--tag", "k2")).name == "k2_classifier.jsonl"          # or a suite to build


# --- 00: the benchmark -------------------------------------------------------------------------------------

def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name in members:
            z.writestr(name, b"not a pickle: 00 only unpacks")
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_get_benchmark_checks_the_md5_unpacks_only_instances_and_refuses_a_member_outside_its_folder(tmp_path, monkeypatch, capsys):
    get = script("00_get_benchmark")
    good = tmp_path / "good.zip"
    monkeypatch.setattr(get, "MD5", make_zip(good, ["instances/", "instances/a.pkl", "instances/b.pkl"]))
    monkeypatch.setattr(get, "N_INSTANCES", 2)
    monkeypatch.setattr(sys, "argv", ["00", "--zip", str(good), "--dest", str(tmp_path / "dest")])
    get.main()
    assert "md5 ok" in capsys.readouterr().out and benchmark.names(tmp_path / "dest" / "instances") == ["a", "b"]
    get.main()                                                                                         # again: the instances are there
    assert "already in" in capsys.readouterr().out
    monkeypatch.setattr(get, "MD5", "0" * 32)
    monkeypatch.setattr(sys, "argv", ["00", "--zip", str(good), "--dest", str(tmp_path / "dest"), "--force"])
    with pytest.raises(SystemExit, match="md5 mismatch"):
        get.main()
    for member in ("instances/../../outside.pkl", "instances/../x.pkl", "elsewhere/a.pkl", "instances/readme.txt"):
        bad = tmp_path / "bad.zip"
        monkeypatch.setattr(get, "MD5", make_zip(bad, [member]))
        monkeypatch.setattr(sys, "argv", ["00", "--zip", str(bad), "--dest", str(tmp_path / "dest2")])
        with pytest.raises(ValueError, match="unexpected member"):
            get.main()
    few = tmp_path / "few.zip"
    monkeypatch.setattr(get, "MD5", make_zip(few, ["instances/a.pkl"]))
    monkeypatch.setattr(sys, "argv", ["00", "--zip", str(few), "--dest", str(tmp_path / "dest3")])
    with pytest.raises(SystemExit, match="expected 2 instances"):
        get.main()


def test_get_benchmark_downloads_when_it_has_no_zip(tmp_path, monkeypatch, capsys):
    get = script("00_get_benchmark")
    source = tmp_path / "source.zip"
    monkeypatch.setattr(get, "MD5", make_zip(source, ["instances/a.pkl"]))
    monkeypatch.setattr(get, "N_INSTANCES", 1)
    monkeypatch.setattr(get.urllib.request, "urlopen", lambda url: nullcontext(io.BytesIO(source.read_bytes())))
    monkeypatch.setattr(sys, "argv", ["00", "--dest", str(tmp_path / "dest")])
    get.main()
    assert "downloading" in capsys.readouterr().out and benchmark.names(tmp_path / "dest" / "instances") == ["a"]


@pytest.mark.parametrize("name", ["00_get_benchmark", "01_build_instances", "02_compute_descriptors", "03_compute_features"])
def test_every_script_runs_as_a_program_and_says_its_usage_with_help(name, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(config.ROOT / "scripts" / f"{name}.py"), run_name="__main__")
    assert stop.value.code == 0 and "usage" in capsys.readouterr().out.lower()


# --- 01, 02, 03: the chain ---------------------------------------------------------------------------------

def test_the_chain_01_02_03_on_a_tiny_suite_writes_three_files_and_a_second_run_computes_nothing(capsys, monkeypatch):
    run("01_build_instances", "--build", "manual", *LATTICE, "--tag", "tiny", monkeypatch=monkeypatch)
    path = config.INSTANCES / "tiny_manual.jsonl"
    assert len(suite.load(path)) == 2 and config.meta_path(path).exists()
    run("02_compute_descriptors", "--suite", str(path), monkeypatch=monkeypatch)
    out = config.descriptors_path(path)
    assert len(descriptors.load(out)) == 2 and out.with_name("tiny_manual_class_table.md").exists() and config.meta_path(out).exists()
    assert "lattice-dim2" in out.with_name("tiny_manual_class_table.md").read_text(encoding="utf-8")
    run("03_compute_features", "--suite", str(path), monkeypatch=monkeypatch)
    out3 = config.features_path(path)
    assert len(features.load(out3)) == 2 and config.meta_path(out3).exists()
    capsys.readouterr()
    run("02_compute_descriptors", "--suite", str(path), monkeypatch=monkeypatch)
    run("03_compute_features", "--suite", str(path), monkeypatch=monkeypatch)
    printed = capsys.readouterr().out
    assert "0 components measured (2 already there)" in printed and "0 components computed (2 already there)" in printed
    run("01_build_instances", "--build", "manual", *LATTICE, "--tag", "tiny", monkeypatch=monkeypatch)
    assert "already there" in capsys.readouterr().out                                                 # the same request: the file is used


def test_02_and_03_stop_on_a_file_made_with_other_settings_and_fresh_starts_again(capsys, monkeypatch):
    run("01_build_instances", "--build", "manual", *LATTICE, "--tag", "tiny", monkeypatch=monkeypatch)
    path = config.INSTANCES / "tiny_manual.jsonl"
    flags = ("--suite", str(path))
    run("02_compute_descriptors", *flags, "--fast", monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match="table_id.*--fresh"):                                       # a file of the fast probes is not the full file
        run("02_compute_descriptors", *flags, monkeypatch=monkeypatch)
    run("02_compute_descriptors", *flags, "--fresh", monkeypatch=monkeypatch)
    assert len(descriptors.load(config.descriptors_path(path))) == 2
    with pytest.raises(ValueError, match="seed.*--fresh"):
        run("02_compute_descriptors", *flags, "--seed", "7", monkeypatch=monkeypatch)
    run("03_compute_features", *flags, monkeypatch=monkeypatch)
    monkeypatch.setattr(features, "FEATURES_ID", "0" * 16)                                           # a change of RWPE_STEPS or of a column
    with pytest.raises(ValueError, match="features_id.*--fresh"):
        run("03_compute_features", *flags, monkeypatch=monkeypatch)
    capsys.readouterr()
    run("03_compute_features", *flags, "--fresh", monkeypatch=monkeypatch)
    assert "2 components computed (0 already there)" in capsys.readouterr().out


def test_02_and_03_resume_with_limit_skip_what_is_too_big_and_measure_the_benchmark_too(tmp_path, capsys, monkeypatch):
    shapes = [(2, 2), (2, 2), (2, 2)]
    with open(tmp_path / "qc_toy.pkl", "wb") as fh:                                                   # a closed triangle in the benchmark's format
        pickle.dump(("ab,bc,ca->", [np.zeros(s, dtype=np.float32) for s in shapes], (([(0, 1), (0, 1)], 1.0, 1.0),) * 2, None), fh, protocol=4)
    insts = [benchmark.load("qc_toy", tmp_path)] + [gen("tree", 16, seed) for seed in range(25)] + [gen("tree", 16, 0, legs="scattered", k=3)]
    path = config.INSTANCES / "many.jsonl"
    suite.save(insts, path)
    flags = ("--suite", str(path), "--fast")
    run("02_compute_descriptors", *flags, "--limit", "1", monkeypatch=monkeypatch)
    assert "1 components measured (0 already there)" in capsys.readouterr().out
    run("02_compute_descriptors", *flags, monkeypatch=monkeypatch)                                    # the rest: 25 trees (the tree with legs is the first tree again)
    printed = capsys.readouterr().out
    assert "25 components  " in printed and "25 components measured (1 already there)" in printed
    assert "benchmark" in config.descriptors_path(path).with_name("many_class_table.md").read_text(encoding="utf-8")
    monkeypatch.setattr(features, "MAX_DENSE", 3)                                                     # the trees have 16 tensors: too big for the dense rwpe
    run("03_compute_features", "--suite", str(path), "--limit", "1", monkeypatch=monkeypatch)
    assert "1 components computed (0 already there), 0 above MAX_DENSE = 3 skipped" in capsys.readouterr().out      # the limit stops after one computed
    run("03_compute_features", "--suite", str(path), monkeypatch=monkeypatch)
    assert "0 components computed (1 already there), 25 above MAX_DENSE = 3 skipped" in capsys.readouterr().out    # a component that is too big is counted again

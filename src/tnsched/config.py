"""Paths, and the memo that goes next to every file the scripts write.

results/ holds what the scripts produce, one folder per script; data/ holds only what comes from
outside (the Einsum Benchmark that scripts/00_get_benchmark.py downloads).
"""
from __future__ import annotations

import datetime
import json
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
# the Einsum Benchmark's instances/*.pkl; set TNSCHED_BENCHMARK to keep them elsewhere
BENCHMARK = Path(os.environ.get("TNSCHED_BENCHMARK", DATA / "einsum_benchmark" / "instances"))
RESULTS = ROOT / "results"
INSTANCES = RESULTS / "instances"          # script 01: <name>.jsonl + <name>.meta.json
DESCRIPTORS = RESULTS / "descriptors"      # script 02: <name>.jsonl + <name>.meta.json, <name>_class_table.md
FEATURES = RESULTS / "features"            # script 03: <name>.npz + <name>.meta.json


def descriptors_path(instances_file: Path) -> Path:
    """The descriptors file of an instances file, named after it: results/instances/small_classifier.jsonl ->
    results/descriptors/small_classifier.jsonl."""
    return DESCRIPTORS / f"{instances_file.stem}.jsonl"


def features_path(instances_file: Path) -> Path:
    """The features file of an instances file: results/instances/small_classifier.jsonl ->
    results/features/small_classifier.npz."""
    return FEATURES / f"{instances_file.stem}.npz"


def json_line(obj: Any) -> str:
    """The one JSON form of every file the package writes, one value per line (without the line break). Three choices:
    compact (no space after "," or ":"), which saves about 15% of the instances file; UTF-8 as it is (cotengra's labels
    reach U+2028 and U+2029 after 8,092 indices, and escaping every non-ASCII character would triple those labels);
    and the three characters that str.splitlines() takes for a line break, U+0085, U+2028 and U+2029, which json leaves
    raw, written as \\u0085, \\u2028, \\u2029, so that no reader or editor cuts a record there. json.loads gives the
    same value back. A value that json cannot write (a Path) is written as text."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return text.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


# --- the meta file: what a file was made with ------------------------------------------------------------
# Other library versions may draw other networks or measure other numbers, so every file written by a script has
# a memo next to it. One writer and one reader, shared by the three scripts.

def meta_path(path: Path) -> Path:
    """reference.jsonl -> reference.meta.json, next to it."""
    return path.with_name(path.stem + ".meta.json")


def library_versions() -> dict[str, str]:
    """The versions that decide what the files hold: python, the libraries that draw, measure or compute, and
    tnsched itself. Read from the installed distributions, so that config imports none of them; one that is
    not installed (tnsched, when run from the source folder) is left out."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "networkx", "cotengra", "tnsched"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def write_meta(path: Path, **fields: Any) -> None:
    """Write the meta of the file `path`: `fields` (what the file is: an id, the flags of the script that makes
    it, as vars(args)), then the library versions and the date. A path in the fields is written as text."""
    meta_path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {**fields, **library_versions(), "date": datetime.date.today().isoformat()}
    with open(meta_path(path), "w", encoding="utf-8") as fh:
        fh.write(json_line(meta) + "\n")


def read_meta(path: Path, advice: str = "", **expected: Any) -> dict[str, Any]:
    """The meta of the file `path`. ValueError naming the file, and every key of `expected` whose value in the
    meta is another, or the meta itself if it is missing; `advice` (what to run) ends the message."""
    if not meta_path(path).exists():
        raise ValueError(f"{path} has no meta file {meta_path(path).name}. {advice}")
    with open(meta_path(path), encoding="utf-8") as fh:
        meta = json.load(fh)
    differ = [f"{key}: the file has {meta.get(key)!r}, now {want!r}" for key, want in expected.items() if meta.get(key) != want]
    if differ:
        raise ValueError(f"{path} was made with other settings ({'; '.join(differ)}). {advice}")
    return meta

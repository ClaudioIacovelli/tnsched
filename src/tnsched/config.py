"""Paths, the JSON-lines format, and the meta file beside every file the scripts write.

results/ holds what the scripts make, one folder per script; data/ holds only what comes from outside (the Einsum
Benchmark, downloaded by scripts/00_get_benchmark.py).
"""
from __future__ import annotations

import datetime
import json
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
# the Einsum Benchmark's instances/*.pkl; set TNSCHED_BENCHMARK to keep them elsewhere
BENCHMARK_DIR = Path(os.environ.get("TNSCHED_BENCHMARK", DATA_DIR / "einsum_benchmark" / "instances"))
RESULTS_DIR = ROOT_DIR / "results"
INSTANCES_DIR = RESULTS_DIR / "instances"          # script 01: <name>.jsonl + <name>.meta.json
DESCRIPTORS_DIR = RESULTS_DIR / "descriptors"      # script 02: <name>.jsonl + <name>.meta.json, <name>_class_table.md
FEATURES_DIR = RESULTS_DIR / "features"            # script 03: <name>.npz + <name>.meta.json


def descriptors_path(instances_path: Path) -> Path:
    """results/instances/<name>.jsonl -> results/descriptors/<name>.jsonl (script 02)."""
    return DESCRIPTORS_DIR / f"{instances_path.stem}.jsonl"


def components_path(instances_path: Path) -> Path:
    """results/instances/<name>.jsonl -> results/descriptors/<name>_components.jsonl (script 02: the connected
    components of each instance)."""
    return DESCRIPTORS_DIR / f"{instances_path.stem}_components.jsonl"


def features_path(instances_path: Path) -> Path:
    """results/instances/<name>.jsonl -> results/features/<name>.npz (script 03)."""
    return FEATURES_DIR / f"{instances_path.stem}.npz"


def json_line(obj: Any) -> str:
    """One value as one line of JSON (without the line break): the form of every file the package writes.

    Compact: no space after "," or ":" (the instances file is about 15% smaller). Non-ASCII characters stay as they
    are: cotengra's labels are Unicode characters and reach U+2028 after 8,092 indices, and escaping them all would
    triple those labels. Only U+0085, U+2028 and U+2029 are escaped: str.splitlines() reads them as line breaks, so a
    reader or an editor would cut the record there. json.loads gives the same value back; a value json cannot write
    (a Path) becomes text."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return text.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def load_lines(path: Path) -> list[Any]:
    """The values of a file written with json_line, one per line, in order."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# --- the meta file: what a file was made with ------------------------------------------------------------
# Other library versions may draw other networks or measure other numbers, so every file a script writes has a meta
# file beside it: what was asked, the library versions, the date.

def meta_path(path: Path) -> Path:
    """reference.jsonl -> reference.meta.json, next to it."""
    return path.with_name(path.stem + ".meta.json")


def library_versions() -> dict[str, str]:
    """{name: version} of python and of every library that decides what the files hold, tnsched included. Read from
    the installed distributions, so config imports none of them; one not installed (torch on a machine without it,
    tnsched run from the source folder) is left out."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "networkx", "cotengra", "torch", "torch_geometric", "tnsched"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def save_meta(path: Path, **fields: Any) -> None:
    """Write the meta file of `path`: `fields` (what the file is: an id, the script's flags as vars(args)), the
    library versions and the date."""
    meta_path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {**fields, **library_versions(), "date": datetime.date.today().isoformat()}
    with open(meta_path(path), "w", encoding="utf-8") as fh:
        fh.write(json_line(meta) + "\n")


def load_meta(path: Path, advice: str = "", **expected: Any) -> dict[str, Any]:
    """The meta of `path`. ValueError if the meta file is missing or a key of `expected` has another value there;
    the message names the file and every such key, and ends with `advice` (what to run)."""
    if not meta_path(path).exists():
        raise ValueError(f"{path} has no meta file {meta_path(path).name}. {advice}")
    with open(meta_path(path), encoding="utf-8") as fh:
        meta = json.load(fh)
    differ = [f"{key}: the file has {meta.get(key)!r}, now {want!r}"
              for key, want in expected.items() if meta.get(key) != want]
    if differ:
        raise ValueError(f"{path} was made with other settings ({'; '.join(differ)}). {advice}")
    return meta

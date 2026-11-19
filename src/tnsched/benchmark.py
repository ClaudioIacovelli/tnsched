"""The Einsum Benchmark: 168 real-world tensor networks, the test set, left as it comes [BSK24].

Blacher et al., NeurIPS 2024 Datasets and Benchmarks, doi:10.5281/zenodo.11477304. The networks come
in 7 categories (gm, lm, mc, qc, rnd, str, wmc); the first part of an instance name is its category
(qc_qft_27: qc). scripts/00_get_benchmark.py downloads them into data/einsum_benchmark/;
config.BENCHMARK says where they are read from.

One pickle holds (format_string, tensors, path_meta, sum_output): the einsum string, the numpy
arrays (only their shapes are read here), the stored paths with their costs, and the output of the
sum (not read). path_meta[0] is the path the authors found for the least size, path_meta[1] the one
for the least FLOPs, each (path, log2 size, log10 flops, min density, avg density). Both are
best-found paths, never optima: a search stopped at some point, with no proof behind it. The keys
"opt_size" and "opt_flops" in params["paths"] are the authors' names for them.

Instances are not redrawn: a disconnected one stays disconnected (Instance.components > 1, 45 of
the 168), and a disconnected network is its components contracted independently. The one change on
loading: five instances (mc_2020_017, _062, _082, mc_2022_167, wmc_2023_141, 68 611 to 349 990
labels) have 1858-2048 labels in U+D800-DFFF, lone UTF-16 surrogates. No file can hold them as
UTF-8, and a JSON reader joins two neighbouring ones into one character, which changes the network.
They are renamed to unused characters (params["relabelled"] counts them); labels are only names,
and paths refer to positions, so the network and its stored paths are unchanged.

The module reads top to bottom: the safe unpickler, names(), then the steps that load() runs one
after the other (_read_pickle, _network, without_surrogates, _stored_paths).
"""
from __future__ import annotations

import pathlib
import pickle
from typing import Any

import cotengra as ctg
import numpy as np

from . import config
from .instances import Inputs, Instance, Output, SizeDict

# path_meta[0] and path_meta[1]: the authors' names for the two stored best-found paths.
STORED_PATHS = ("opt_size", "opt_flops")

# The code points D800-DFFF are reserved for UTF-16 surrogate halves; alone, one is not a character.
SURROGATES = range(0xD800, 0xE000)

# The only globals a benchmark pickle may name: what numpy needs to rebuild an array.
_ALLOWED_GLOBALS = {("numpy.core.multiarray", "_reconstruct"),      # numpy 1 spelling
                    ("numpy._core.multiarray", "_reconstruct"),     # numpy 2 spelling
                    ("numpy", "ndarray"), ("numpy", "dtype")}


# --- reading a pickle safely ---------------------------------------------------------------

# A .pkl file is Python objects written out as bytes (the pickle module). Reading one does more
# than read data: the stream can tell Python to import any function and call it, so loading a file
# from the internet can run arbitrary code. The only hook Python gives for that is find_class:
# Unpickler.load() calls it for EVERY global (module.name: a class or a function) the stream
# names, and uses whatever it returns. Overriding it to refuse all but a short list makes the
# read safe.
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Loads the benchmark's pickles, refusing every global except the three numpy needs to
    rebuild an array."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        if name == "_reconstruct":
            # Pickles written under numpy 1 name numpy.core, under numpy 2 numpy._core; both are
            # served from np._core, which exists from numpy 2.0 on (Python 3.14 needs numpy >= 2).
            return np._core.multiarray._reconstruct
        return np.ndarray if name == "ndarray" else np.dtype


def names(root: pathlib.Path = config.BENCHMARK) -> list[str]:
    """The instance names: "qc_qft_27.pkl" -> "qc_qft_27", sorted."""
    return sorted(p.stem for p in root.glob("*.pkl"))


def _read_pickle(path: pathlib.Path) -> tuple[str, list[Any], tuple]:
    """(format_string, tensors, path_meta) of one benchmark file, read with the safe unpickler.
    ValueError (from the unpacking) if the file does not hold the four fields (format_string,
    tensors, path_meta, sum_output)."""
    with open(path, "rb") as fh:                                  # "rb": a pickle is bytes, not text
        format_string, tensors, path_meta, _ = _NumpyOnlyUnpickler(fh).load()
    return format_string, tensors, path_meta


# --- the network ---------------------------------------------------------------------------

def _network(format_string: str, shapes: list[tuple[int, ...]]) -> tuple[Inputs, Output, SizeDict]:
    """The triple (inputs, output, size_dict) of an einsum string and the shapes of its tensors."""
    # eq_to_inputs_output splits the string at "->" and at the commas:
    # "ab,bc->ac" -> ([("a","b"), ("b","c")], ("a","c")); each term comes as a sequence of labels.
    terms, open_labels = ctg.utils.eq_to_inputs_output(format_string)
    inputs = [tuple(term) for term in terms]                      # tuples: the Network type
    # shapes_inputs_to_size_dict reads the dimension of every label off the shapes: if tensor 0 is
    # "ab" with shape (2, 3), then a = 2 and b = 3. A label seen twice must agree.
    found = ctg.utils.shapes_inputs_to_size_dict(shapes, inputs)
    return inputs, tuple(open_labels), {label: int(d) for label, d in found.items()}   # a plain int, not a numpy one


def _renamed(term: tuple[str, ...], new: dict[str, str]) -> tuple[str, ...]:
    """The labels of one tensor with the renamed ones replaced and the others kept."""
    return tuple(new.get(label, label) for label in term)


def without_surrogates(inputs: Inputs, output: Output,
                       size_dict: SizeDict) -> tuple[Inputs, Output, SizeDict, int]:
    """Rename every label that holds a lone surrogate to an unused character above all labels in
    use (and above U+DFFF); the others keep their names. Returns the network and how many labels
    were renamed. Labels are only names, so nothing else changes."""
    bad = sorted(label for label in size_dict if any(ord(char) in SURROGATES for char in label))
    if not bad:
        return inputs, output, size_dict, 0
    # The new names are the characters from `start` on: one above every code point in use, and
    # above U+DFFF, so that no new name is a surrogate or collides with an existing label.
    start = max(SURROGATES[-1], max(ord(char) for label in size_dict for char in label)) + 1
    new = {label: chr(start + k) for k, label in enumerate(bad)}   # chr: the code point -> its character
    new_sizes = {new.get(label, label): d for label, d in size_dict.items()}
    return [_renamed(term, new) for term in inputs], _renamed(output, new), new_sizes, len(bad)


# --- the stored paths ----------------------------------------------------------------------

def _stored_paths(path_meta: tuple) -> dict[str, dict[str, Any]]:
    """The authors' two stored paths, keyed opt_size and opt_flops (STORED_PATHS), each as
    {"path", "size", "flops"}. path_meta[i] is (path, log2 size, log10 flops, densities...);
    a step is the positions of the tensors it contracts, integers in the pickle. ValueError if the
    pickle holds fewer than two paths."""
    if len(path_meta) < len(STORED_PATHS):
        raise ValueError(f"expected {len(STORED_PATHS)} stored paths, the pickle has {len(path_meta)}")
    return {key: {"path": [tuple(int(position) for position in step) for step in meta[0]],
                  "size": float(meta[1]),     # log2 of the largest intermediate (entries)
                  "flops": float(meta[2])}    # log10 (not log2!) of opt_einsum's FLOP count
            for key, meta in zip(STORED_PATHS, path_meta)}


# --- the pipeline --------------------------------------------------------------------------

def load(name: str, root: pathlib.Path = config.BENCHMARK) -> Instance:
    """One benchmark instance (family "benchmark", seed None), with its two stored best-found paths
    (keys opt_size, opt_flops, after the authors; not optima) in params["paths"]."""
    format_string, tensors, path_meta = _read_pickle(root / f"{name}.pkl")
    # Only the shapes matter here, and the arrays are big (all 168 files hold 1.3 GB): keep the
    # shapes and let go of the arrays at once, so that they are freed before the rest runs.
    shapes = [tensor.shape for tensor in tensors]
    del tensors
    inputs, output, size_dict = _network(format_string, shapes)
    inputs, output, size_dict, relabelled = without_surrogates(inputs, output, size_dict)
    params = {"n": len(inputs), "name": name, "category": name.split("_")[0],
              "paths": _stored_paths(path_meta), "relabelled": relabelled}
    return Instance(inputs, output, size_dict, "benchmark", params, None)

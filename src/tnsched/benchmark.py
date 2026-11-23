"""The Einsum Benchmark [BSK24]: 168 real-world tensor networks, taken as they come.

Blacher et al., NeurIPS 2024 Datasets and Benchmarks, doi:10.5281/zenodo.11477304. 7 categories (gm, lm, mc, qc, rnd,
str, wmc), the first part of a name (qc_qft_27: qc). scripts/00_get_benchmark.py downloads them; config.BENCHMARK_DIR
says where they are read from.

One pickle holds (format_string, tensors, path_meta, sum_output): the einsum string, the arrays (only their shapes
are read), two stored paths, and the result (not read). path_meta[0] is the authors' path for the least size,
path_meta[1] for the least FLOPs, each (path, log2 size, log10 flops, min density, avg density). Their keys
"opt_size" and "opt_flops" are the authors' names: both are best found, not proven optima.

Nothing is redrawn: 45 of the 168 are disconnected (Instance.components > 1). One change on loading: five networks
(mc_2020_017, _062, _082, mc_2022_167, wmc_2023_141) have 1858-2048 labels that are lone UTF-16 surrogates
(U+D800-DFFF). UTF-8 cannot hold them, and a JSON reader joins two neighbours into one character, which changes the
network. They are renamed to unused characters (params["relabelled"] counts them); paths refer to positions, so the
network and its paths are unchanged.

Reading order: the safe unpickler, names(), then the steps of load() in order (_read_pickle, _network,
without_surrogates, _stored_paths).
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

# A .pkl file is Python objects written as bytes. Reading one can import and call any function the stream names, so
# a downloaded file can run arbitrary code. Unpickler.load() asks find_class for EVERY global (module.name, a class
# or a function) the stream names: refusing all but a short list makes the read safe.
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """An unpickler that refuses every global except what numpy needs to rebuild an array (_ALLOWED_GLOBALS)."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        if name == "_reconstruct":
            # numpy 1 wrote numpy.core, numpy 2 numpy._core; both are served from np._core (numpy >= 2)
            return np._core.multiarray._reconstruct
        return np.ndarray if name == "ndarray" else np.dtype


def names(root: pathlib.Path = config.BENCHMARK_DIR) -> list[str]:
    """The instance names: "qc_qft_27.pkl" -> "qc_qft_27", sorted."""
    return sorted(p.stem for p in root.glob("*.pkl"))


def _read_pickle(path: pathlib.Path) -> tuple[str, list[Any], tuple]:
    """(format_string, tensors, path_meta) of one benchmark file, read with the safe unpickler. ValueError (from the
    unpacking) if the file does not hold four fields."""
    with open(path, "rb") as fh:                                  # "rb": a pickle is bytes, not text
        format_string, tensors, path_meta, _ = _NumpyOnlyUnpickler(fh).load()
    return format_string, tensors, path_meta


# --- the network ---------------------------------------------------------------------------

def _network(format_string: str, shapes: list[tuple[int, ...]]) -> tuple[Inputs, Output, SizeDict]:
    """The triple (inputs, output, size_dict) of an einsum string and the shapes of its tensors."""
    # split at "->" and at the commas: "ab,bc->ac" -> ([("a","b"), ("b","c")], ("a","c"))
    terms, open_labels = ctg.utils.eq_to_inputs_output(format_string)
    inputs = [tuple(term) for term in terms]                      # tuples: the Network type
    # the dimension of every label off the shapes: tensor "ab" of shape (2, 3) gives a = 2, b = 3
    found = ctg.utils.shapes_inputs_to_size_dict(shapes, inputs)
    return inputs, tuple(open_labels), {label: int(d) for label, d in found.items()}   # a plain int, not a numpy one


def _renamed(term: tuple[str, ...], new: dict[str, str]) -> tuple[str, ...]:
    """The labels of one tensor with the renamed ones replaced and the others kept."""
    return tuple(new.get(label, label) for label in term)


def without_surrogates(inputs: Inputs, output: Output,
                       size_dict: SizeDict) -> tuple[Inputs, Output, SizeDict, int]:
    """(inputs, output, size_dict, renamed): every label holding a lone surrogate renamed to an unused character;
    renamed counts them. Labels are only names: nothing else changes."""
    bad = sorted(label for label in size_dict if any(ord(char) in SURROGATES for char in label))
    if not bad:
        return inputs, output, size_dict, 0
    # new names from `start` on: above every code point in use and above U+DFFF, so none is a surrogate or taken
    start = max(SURROGATES[-1], max(ord(char) for label in size_dict for char in label)) + 1
    new = {label: chr(start + k) for k, label in enumerate(bad)}   # chr: the code point -> its character
    new_sizes = {new.get(label, label): d for label, d in size_dict.items()}
    return [_renamed(term, new) for term in inputs], _renamed(output, new), new_sizes, len(bad)


# --- the stored paths ----------------------------------------------------------------------

def _stored_paths(path_meta: tuple) -> dict[str, dict[str, Any]]:
    """{"opt_size" | "opt_flops": {"path", "size", "flops"}}: the authors' two paths. A step is the pair of positions
    it contracts. ValueError if the pickle holds fewer than two paths."""
    if len(path_meta) < len(STORED_PATHS):
        raise ValueError(f"expected {len(STORED_PATHS)} stored paths, the pickle has {len(path_meta)}")
    return {key: {"path": [tuple(int(position) for position in step) for step in meta[0]],
                  "size": float(meta[1]),     # log2 of the largest intermediate (entries)
                  "flops": float(meta[2])}    # log10 (not log2!) of opt_einsum's FLOP count
            for key, meta in zip(STORED_PATHS, path_meta)}


# --- the pipeline --------------------------------------------------------------------------

def load(name: str, root: pathlib.Path = config.BENCHMARK_DIR) -> Instance:
    """One benchmark instance: family "benchmark", seed None, its two stored paths in params["paths"]."""
    format_string, tensors, path_meta = _read_pickle(root / f"{name}.pkl")
    # only the shapes matter, and the arrays are big (1.3 GB in all): free them at once
    shapes = [tensor.shape for tensor in tensors]
    del tensors
    inputs, output, size_dict = _network(format_string, shapes)
    inputs, output, size_dict, relabelled = without_surrogates(inputs, output, size_dict)
    params = {"n": len(inputs), "name": name, "category": name.split("_")[0],
              "paths": _stored_paths(path_meta), "relabelled": relabelled}
    return Instance(inputs, output, size_dict, "benchmark", params, None)

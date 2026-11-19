"""Compute the model's input of every connected component of an instances file; write results/features/<name>.npz.

For every component of every instance (descriptors.split(inst.core, 0)): features.component_input, the native
node columns and the rwpe columns, keyed by the component's id. rwpe is dense (up to 64 products of T x T matrices,
T <= features.MAX_DENSE): it is computed once and kept in this file. A component above MAX_DENSE tensors is skipped
and counted. The ids already in the file are skipped; its meta records features.FEATURES_ID and the flags of this
script, and a file made with another FEATURES_ID is an error: after a change of features.FEATURES, RWPE_STEPS or the
constants, run again with --fresh. The file is written once, at the end. The file is named after the instances file
(config.features_path). Which instances file: the menu of _shared.py (--suite, --build, or a question).

Usage: python scripts/03_compute_features.py [--suite PATH | --build reference|classifier|manual ...]
                                             [--out PATH] [--fresh] [--limit N]
"""
import argparse
import time
from pathlib import Path

from _shared import add_arguments, new_components, pick_instances

from tnsched import config, features, suite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    ap.add_argument("--out", type=Path, default=None, help="default: config.features_path of the instances file")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite what is in --out")
    ap.add_argument("--limit", type=int, default=None, help="stop after N new components (a test run)")
    args = ap.parse_args()

    path = pick_instances(args)
    out = args.out or config.features_path(path)
    if args.fresh:
        out.unlink(missing_ok=True)
        config.meta_path(out).unlink(missing_ok=True)
    first = not out.exists()
    done = {} if first else features.load(out)
    t0 = time.perf_counter()
    new, skipped = [], 0
    for c in new_components(suite.load(path), 0, set(done)):
        if len(new) == args.limit:
            break
        if c.n > features.MAX_DENSE:
            skipped += 1
        else:
            new.append(features.component_input(c))
    if new or first:
        features.save(out, [*done.values(), *new])
    if first:
        config.write_meta(out, features_id=features.FEATURES_ID, flags=vars(args))
    print(f"{len(new)} components computed ({len(done)} already there), {skipped} above MAX_DENSE = "
          f"{features.MAX_DENSE} skipped -> {out} ({out.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()

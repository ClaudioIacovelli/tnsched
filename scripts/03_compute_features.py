"""Compute the model's input of every connected component of an instances file; write results/features/<name>.npz.

Which instances file: --suite, --build, or a question (_shared.py). For every component (descriptors.split(inst.core,
0)): features.component_input, keyed by the component's id. rwpe is dense (up to 64 products of T x T matrices), so it
is computed once and kept here; a component above features.RWPE_MAX_TENSORS tensors is skipped and counted. The ids
already in the file are skipped. The meta records features.FEATURES_ID: after a change of the columns or their
constants the old file is an error, run again with --fresh. The file is written once, at the end.

Usage: python scripts/03_compute_features.py [--suite PATH | --build reference|classifier|manual ...]
                                             [--out PATH] [--fresh] [--limit N]
"""
import argparse
import time
from pathlib import Path

from _shared import add_arguments, new_components, choose_instances_path

from tnsched import config, features, suite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    ap.add_argument("--out", type=Path, default=None, help="default: config.features_path of the instances file")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite what is in --out")
    ap.add_argument("--limit", type=int, default=None, help="stop after N new components (a test run)")
    args = ap.parse_args()

    instances_path = choose_instances_path(args)
    out = args.out or config.features_path(instances_path)
    if args.fresh:
        out.unlink(missing_ok=True)
        config.meta_path(out).unlink(missing_ok=True)
    first = not out.exists()
    done = {} if first else features.load(out)
    t0 = time.perf_counter()
    new, skipped = [], 0
    for c in new_components(suite.load(instances_path), 0, set(done)):
        if len(new) == args.limit:
            break
        if c.n > features.RWPE_MAX_TENSORS:
            skipped += 1
        else:
            new.append(features.component_input(c))
    if new or first:
        features.save(out, [*done.values(), *new])
    if first:
        config.save_meta(out, features_id=features.FEATURES_ID, flags=vars(args))
    print(f"{len(new)} components computed ({len(done)} already there), {skipped} above RWPE_MAX_TENSORS = "
          f"{features.RWPE_MAX_TENSORS} skipped -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()

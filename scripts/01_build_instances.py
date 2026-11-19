"""Build an instance suite and write it as JSON lines (structure only, no arrays).

The suites are suite.SUITES: "reference" (the test set: seeds suite.SEEDS, five leg recipes), "classifier" (the
training set: closed networks, other seeds), or "manual" (one family, from the flags). The file is
results/instances/<suite>.jsonl with its meta file; if it is already there, a tag is asked for (or --tag), and
Enter uses the file that is there. --benchmark adds the 168 Einsum Benchmark networks.

Usage: python scripts/01_build_instances.py [--build reference|classifier|manual] [--tag TAG] [--benchmark]
                                            [--seeds 0 1 2] [--bonds uniform] [--max-n N] [--chi 2]
       manual: --family F --knobs '{"dim": 2}' --sizes 16 64 --seeds 0 1
               (Windows PowerShell 5.1 drops the inner quotes: --knobs '{\"dim\": 2}')
"""
import argparse

from _shared import add_arguments, build_instances


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    args = ap.parse_args()
    build_instances(args.build or "reference", args)


if __name__ == "__main__":
    main()

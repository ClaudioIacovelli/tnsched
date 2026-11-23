"""What scripts 01, 02 and 03 share: which instances file to work on, and which components are left to compute.

Every question a script asks is also a flag, so no script needs a person at the keyboard. Python puts a script's own
folder first on sys.path, so `from _shared import ...` works from any directory.
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from tnsched import config, descriptors, instances, suite
from tnsched.descriptors import ConnectedComponent


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """The flags that choose the instances file: an existing one (--suite), or one to build (--build and the rest)."""
    ap.add_argument("--suite", type=Path, help="an instances file to use")
    ap.add_argument("--build", choices=[*suite.SUITES, "manual"], help="build this suite (or use its file)")
    ap.add_argument("--tag", help="a name for the file: <tag>_<suite>.jsonl")
    ap.add_argument("--benchmark", action="store_true",
                    help="add the 168 Einsum Benchmark instances (name <suite>_benchmark)")
    ap.add_argument("--chi", type=int, default=instances.CHI)
    ap.add_argument("--seeds", type=int, nargs="+", help="replace the seeds of the suite (manual: the seeds)")
    ap.add_argument("--bonds", nargs="+", choices=instances.BOND_RULES, help="replace the bond rules of the suite")
    ap.add_argument("--max-n", type=int, help="drop the sizes above this n")
    ap.add_argument("--family", help="manual: the family")
    ap.add_argument("--knobs", help='manual: every knob of the family, as JSON, e.g. \'{"dim": 2}\'')
    ap.add_argument("--sizes", type=int, nargs="+", help="manual: the sizes n")


def _ask(question: str) -> str:
    """The answer to a question, or "" when nobody can answer (stdin is closed)."""
    try:
        return input(question).strip()
    except EOFError:
        return ""


def _suite_of(choice: str, args: argparse.Namespace) -> suite.Suite:
    """A SUITES entry with the flags laid over it, or for "manual" the one-member closed suite of the flags."""
    if choice == "manual":
        missing = [f"--{f}" for f in ("family", "knobs", "sizes", "seeds") if getattr(args, f) is None]
        if missing:
            raise SystemExit(f"--build manual needs {', '.join(missing)}")
        try:
            knobs = json.loads(args.knobs)
        except json.JSONDecodeError:
            knobs = None
        if not isinstance(knobs, dict):
            raise SystemExit(f"--knobs must be a JSON object with every knob of the family, for example "
                             f"--knobs '{{\"dim\": 2}}' (Windows PowerShell 5.1: --knobs '{{\\\"dim\\\": 2}}'); "
                             f"got {args.knobs!r}")
        member = suite.Member(args.family, knobs, tuple(args.sizes))
        return suite.Suite((member,), tuple(args.seeds), tuple(args.bonds or suite.REFERENCE_BONDS), {"closed": ()})
    base = suite.SUITES[choice]
    members = tuple(m._replace(sizes=tuple(n for n in m.sizes if args.max_n is None or n <= args.max_n))
                    for m in base.members)
    return base._replace(members=tuple(m for m in members if m.sizes), seeds=tuple(args.seeds or base.seeds),
                         bonds=tuple(args.bonds or base.bonds))


def _request_of(chosen: suite.Suite, args: argparse.Namespace) -> dict:
    """What a build asks for, in plain JSON types: the suite, chi and whether the benchmark is added. It goes into
    the meta of the file, which is reused only for the same request."""
    request = {"members": [m._asdict() for m in chosen.members], "seeds": chosen.seeds, "bonds": chosen.bonds,
               "legs": chosen.legs, "chi": args.chi, "benchmark": args.benchmark}
    return json.loads(json.dumps(request))


def build_instances(choice: str, args: argparse.Namespace) -> Path:
    """Build the suite `choice` ("reference", "classifier" or "manual") into results/instances/<name>.jsonl and
    return the path.

    The name is the suite's, plus the local date and time for a manual build or a suite changed by --max-n, --seeds
    or --bonds; --tag goes in front instead of the time; --benchmark adds _benchmark. A file already there is reused
    if its meta has the same request; otherwise the script asks for a tag (or stops, without a keyboard)."""
    chosen = _suite_of(choice, args)
    request = _request_of(chosen, args)
    changed = choice == "manual" or any(getattr(args, flag) is not None for flag in ("max_n", "seeds", "bonds"))
    stamp = f"_{datetime.datetime.now():%Y%m%d-%H%M%S}" if changed and not args.tag else ""   # a tag names it instead
    stem = choice + stamp + ("_benchmark" if args.benchmark else "")
    path = config.INSTANCES_DIR / (f"{args.tag}_{stem}.jsonl" if args.tag else f"{stem}.jsonl")
    while path.exists():
        try:
            what = config.load_meta(path, **request)
        except ValueError as error:
            question = f"{error}\nA tag for another file (<tag>_{path.name}), or Enter to stop: "
            tag = _ask(question) if sys.stdin.isatty() else ""
            if not tag:
                raise SystemExit(f"{error} Pass --tag NAME to write another file.")
            path = path.with_name(f"{tag}_{path.name}")
        else:
            print(f"{path.name} is already there, made for this request ({what['n_instances']} instances): using it")
            return path
    t0 = time.perf_counter()
    insts = suite.build(chosen, args.chi, with_benchmark=args.benchmark)
    suite.save(insts, path)
    request_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    config.save_meta(path, request_id=request_id, n_instances=len(insts), flags=vars(args), **request)
    counts = {}                                   # (member, legs) -> how many instances
    for i in insts:
        key = (i.member, i.params.get("legs", "-"))
        counts[key] = counts.get(key, 0) + 1
    for (member, legs), m in sorted(counts.items()):
        print(f"{member:32s} {legs:10s} {m:5d}")
    print(f"{len(insts)} instances -> {path} and {config.meta_path(path).name} ({time.perf_counter() - t0:.1f} s)")
    return path


def choose_instances_path(args: argparse.Namespace) -> Path:
    """The instances file to work on: --suite PATH; else --build CHOICE (build_instances); else list
    results/instances/*.jsonl and ask which, or to build one."""
    if args.suite:
        if not args.suite.exists():
            raise SystemExit(f"{args.suite} does not exist: run scripts/01_build_instances.py, or pass --build")
        return args.suite
    if args.build:
        return build_instances(args.build, args)
    files = sorted(config.INSTANCES_DIR.glob("*.jsonl"))
    if not sys.stdin.isatty():
        choices = ",".join([*suite.SUITES, "manual"])
        raise SystemExit(f"pass --suite PATH or --build {{{choices}}}; files there: {[f.name for f in files]}")
    for k, f in enumerate(files, 1):
        print(f"  {k}) {f.name}")
    answer = _ask(f"which file (1-{len(files)}), or a suite to build ({', '.join([*suite.SUITES, 'manual'])}): ")
    if answer.isdigit() and 1 <= int(answer) <= len(files):
        return files[int(answer) - 1]
    if answer not in (*suite.SUITES, "manual"):
        raise SystemExit(f"{answer!r} is neither a file number nor a suite")
    return build_instances(answer, args)


def forget_graphs(inst: instances.Instance) -> None:
    """Drop the cached graph and core of an instance (a cached_property lives in __dict__), so that a script
    walking a whole file keeps no graph after the instance's turn."""
    inst.__dict__.pop("core", None)
    inst.__dict__.pop("graph", None)


def new_components(insts: list[instances.Instance], seed: int, done: set[str]) -> Iterator[ConnectedComponent]:
    """Every distinct connected component of `insts` (descriptors.split(inst.core, seed)) whose id is not in
    `done`, each once.

    A generator (`yield` hands out one component and resumes on the next request): one component lives at a time,
    because the views are large and the classifier's 4305 components do not fit in memory together."""
    seen = set(done)
    for inst in insts:
        pieces = descriptors.split(inst.core, seed)
        forget_graphs(inst)                     # the pieces are copies: the instance needs no graph while they are used
        for c in pieces:
            if c.id not in seen:
                seen.add(c.id)
                yield c

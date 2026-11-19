"""Compute the structural descriptors of every component of an instances file; write results/descriptors/<name>.jsonl and the class table.

One row per distinct component id (descriptors.measure), appended as soon as it is measured, so a run that stops
keeps what it finished: the next run skips the ids already in the file. The file is named after the instances
file (config.descriptors_path) and its meta records the table of the probes that ran (all, or the fast ones with
--fast), the seed of descriptors.split and the flags of this script: a file made with another table or seed is an
error, --fresh starts again. Which instances file: the menu of _shared.py (--suite, --build, or a question). The
class table (median per member and size of each row on the largest component of the closed instances) is printed
and written to <name>_class_table.md next to the descriptors file. The benchmark is measured too if the instances
file holds it (01 --benchmark).

Usage: python scripts/02_compute_descriptors.py [--suite PATH | --build reference|classifier|manual ...]
                                                [--out PATH] [--fresh] [--fast] [--seed 0] [--limit N]
"""
import argparse
import itertools
import statistics
import time
from pathlib import Path

from _shared import add_arguments, forget_graphs, new_components, pick_instances

from tnsched import config, descriptors, suite

TABLE_COLUMNS = tuple(descriptors.ROWS)                          # every row of the table, a group's rows included


def class_table(rows: dict, parts: list[tuple[str, int | None, str]], theory: dict[str, float]) -> str:
    """Markdown: per member and size, the median of each row over the seeds. `parts` has one entry per closed
    instance: (member, n, id of its largest component); the benchmark has no n."""
    cells = {}                                  # a cell: (member, n) -> column -> the values over the seeds
    for member, n, component_id in parts:
        if component_id not in rows:
            continue
        columns = cells.setdefault((member, n), {})
        for column in TABLE_COLUMNS:
            if rows[component_id].get(column) is not None:
                columns.setdefault(column, []).append(float(rows[component_id][column]))
    lines = ["| member | n | " + " | ".join(TABLE_COLUMNS) + " | alpha theory |",
             "|" + "---|" * (len(TABLE_COLUMNS) + 3)]
    for (member, n), columns in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        values = [f"{statistics.median(columns[c]):.3g}" if c in columns else "-" for c in TABLE_COLUMNS]
        lines.append(f"| {member} | {'-' if n is None else n} | " + " | ".join(values) + f" | {theory.get(member, '-')} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    ap.add_argument("--out", type=Path, default=None, help="default: config.descriptors_path of the instances file")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite what is in --out")
    ap.add_argument("--fast", action="store_true", help="only the fast descriptors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="stop after N new components (a test run)")
    args = ap.parse_args()

    path = pick_instances(args)
    out = args.out or config.descriptors_path(path)
    if args.fresh:
        out.unlink(missing_ok=True)
        config.meta_path(out).unlink(missing_ok=True)
    done = descriptors.load(out, not args.fast, args.seed) if out.exists() else {}
    if not out.exists():
        config.write_meta(out, table_id=descriptors.table_id(not args.fast), seed=args.seed, flags=vars(args))
    insts = suite.load(path)
    t0 = time.perf_counter()
    k = 0
    for k, c in enumerate(itertools.islice(new_components(insts, args.seed, set(done)), args.limit), 1):
        descriptors.save(out, descriptors.measure(c, slow=not args.fast))
        if k % 25 == 0:
            print(f"{k} components  {time.perf_counter() - t0:.0f} s", flush=True)
    # the "alpha theory" column exists for the reference members only: found by family and knobs
    alpha = {(m.family, tuple(sorted(m.knobs.items()))): round(m.alpha, 2) for m in suite.MEMBERS}
    theory, parts = {}, []
    for inst in insts:
        key = (inst.family, tuple(sorted(inst.params.get("knobs", {}).items())))
        if key in alpha:
            theory[inst.member] = alpha[key]
        if not inst.output:
            parts.append((inst.member, None if inst.family == "benchmark" else inst.params["n"],
                          descriptors.split(inst.core, args.seed)[0].id))
        forget_graphs(inst)
    table = class_table(descriptors.load(out, not args.fast, args.seed) if out.exists() else {}, parts, theory)
    out.with_name(f"{out.stem}_class_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"{k} components measured ({len(done)} already there), {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()

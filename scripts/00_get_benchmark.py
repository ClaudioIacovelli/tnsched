"""Get the Einsum Benchmark: download instances.zip from Zenodo, check its md5, unzip.

The 168 instances (552 MB zipped, 1.3 GB unpacked) go to data/einsum_benchmark/instances/, where
config.BENCHMARK_DIR reads them. --zip uses an instances.zip already on disk instead, checked the same way.

Usage: python scripts/00_get_benchmark.py [--zip PATH] [--dest DIR] [--force]
"""
import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tnsched import benchmark, config

URL = "https://zenodo.org/records/11477304/files/instances.zip?download=1"   # doi:10.5281/zenodo.11477304
MD5 = "b476f37d39a5b8a978e8230375326eb3"                     # what the Zenodo record lists for the zip
N_INSTANCES = 168                                            # and what it promises to hold


def unzip(archive: Path, dest: Path) -> None:
    """Extract instances/*.pkl into dest; ValueError for any other member (a zip can name paths outside dest)."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            target = (dest / member).resolve()
            if target.parent != dest / "instances" or target.suffix != ".pkl":
                raise ValueError(f"unexpected member in {archive.name}: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, help="use this instances.zip instead of downloading")
    ap.add_argument("--dest", type=Path, default=config.BENCHMARK_DIR.parent)
    ap.add_argument("--force", action="store_true", help="unzip even if the instances are there")
    args = ap.parse_args()

    have = len(benchmark.names(args.dest / "instances"))
    if have == N_INSTANCES and not args.force:
        print(f"{have} instances already in {args.dest / 'instances'}; --force to unzip again")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = args.zip
        if archive is None:
            archive = Path(tmp) / "instances.zip"
            print(f"downloading {URL}")
            with urllib.request.urlopen(URL) as response, open(archive, "wb") as fh:
                shutil.copyfileobj(response, fh)
            print(f"{archive.stat().st_size / 2**20:.1f} MB")
        with open(archive, "rb") as fh:
            got = hashlib.file_digest(fh, "md5").hexdigest()
        if got != MD5:
            raise SystemExit(f"md5 mismatch: {got}, expected {MD5}; the file is not the benchmark")
        print(f"md5 ok ({MD5})")
        unzip(archive, args.dest)
    have = len(benchmark.names(args.dest / "instances"))
    print(f"{have} instances in {args.dest / 'instances'}")
    if have != N_INSTANCES:
        raise SystemExit(f"expected {N_INSTANCES} instances")


if __name__ == "__main__":
    main()

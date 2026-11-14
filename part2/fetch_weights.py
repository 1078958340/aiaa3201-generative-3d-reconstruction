"""Fetch or verify the exact checkpoints used in the Part 2 experiments."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def ensure_file(destination, record, check_only):
    expected = record["sha256"]
    if destination.exists():
        if sha256(destination) not in (expected, record.get("archived_sha256")):
            raise ValueError(f"SHA-256 mismatch: {destination}. Save or remove this file before replacing it.")
        print(f"Verified {destination.name}", flush=True)
        return
    if check_only:
        raise FileNotFoundError(f"Missing {destination}. Run part2/fetch_weights.py to install the weights.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(record["url"], headers={"User-Agent": "AIAA3201-Part2"})
    print(f"Downloading {destination.name}", flush=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            copyfileobj(response, handle, length=1024 * 1024)
        if sha256(temporary) != expected:
            raise ValueError(f"Unexpected SHA-256 for the download of {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "reconstruct", "evaluate"), default="all")
    parser.add_argument("--check", action="store_true", help="Verify files already on disk.")
    args = parser.parse_args()

    manifest = json.loads(Path(__file__).with_name("weights.json").read_text(encoding="utf-8"))
    work = Path(os.environ["P2_WORK"])
    roots = {"weights": work / "weights", "source": Path(os.environ.get("P2_SRC", work / "i3dgs"))}
    for record in manifest:
        if args.stage not in ("all", record["stage"]):
            continue
        path = roots[record["root"]] / record["path"]
        ensure_file(path, record, args.check)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None

#!/usr/bin/env python3
"""Select a deterministic, real-valued SuiteSparse performance corpus.

The official ``ssstats.csv`` supplies reproducible metadata. Selection is
stratified by log10(nnz), then diversified by average row length, structural
kind, 2D/3D classification, symmetry and positive-definiteness. If matrix
archives are available locally, ``--inspect-root`` adds measured row-length
variance to the diversity score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
FIELDS = ("group", "name", "rows", "cols", "nnz", "real", "binary", "is2d3d",
          "posdef", "pattern_symmetry", "numerical_symmetry", "kind", "id")


def load_index(opener, path: Path | None) -> tuple[list[dict[str, str]], str]:
    if path:
        raw = path.read_bytes()
    else:
        raw = opener.open(Request(INDEX_URL, headers={"User-Agent": "QiWu-KernelPerf/1.0"}), timeout=60).read()
    lines = raw.decode("utf-8-sig").splitlines()
    entries = list(csv.DictReader(lines[2:], fieldnames=FIELDS))
    return entries, hashlib.sha256(raw).hexdigest()


def log_bin(nnz: int) -> int:
    return min(7, max(2, int(math.log10(nnz))))


def row_variance(path: Path) -> float | None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = [item for item in archive.getmembers() if item.isfile() and item.name.lower().endswith(".mtx")]
            if len(members) != 1:
                return None
            stream = archive.extractfile(members[0])
            if stream is None:
                return None
            text = io.TextIOWrapper(stream, encoding="utf-8", errors="strict")
            header = text.readline().lower().split()
            if len(header) < 5 or header[2] != "coordinate":
                return None
            symmetry = header[4]
            line = text.readline()
            while line.startswith("%"):
                line = text.readline()
            rows, _, count = (int(value) for value in line.split()[:3])
            lengths = [0] * rows
            for _ in range(count):
                fields = text.readline().split()
                if len(fields) >= 2:
                    row = int(fields[0]) - 1
                    lengths[row] += 1
                    if symmetry in {"symmetric", "hermitian", "skew-symmetric"}:
                        col = int(fields[1]) - 1
                        if row != col and 0 <= col < rows:
                            lengths[col] += 1
            mean = sum(lengths) / rows
            return sum((value - mean) ** 2 for value in lengths) / rows
    except (OSError, tarfile.TarError, ValueError, UnicodeError):
        return None


def select(entries: list[dict[str, str]], count: int, inspect_root: Path | None) -> list[dict[str, object]]:
    candidates = []
    for raw in entries:
        try:
            nnz = int(raw["nnz"])
            rows = int(raw["rows"])
            cols = int(raw["cols"])
        except (KeyError, ValueError):
            continue
        if raw.get("real") != "1" or not (100 <= nnz <= 100_000_000) or rows <= 0 or cols <= 0:
            continue
        item = {
            "matrix_id": f"{raw['group']}/{raw['name']}",
            "name": raw["name"],
            "group": raw["group"],
            "rows": rows,
            "cols": cols,
            "nnz": nnz,
            "source_url": f"https://sparse.tamu.edu/MM/{raw['group']}/{raw['name']}.tar.gz",
            "real": True,
            "pattern": {
                "kind": raw.get("kind", ""),
                "is2d3d": raw.get("is2d3d", ""),
                "posdef": raw.get("posdef", ""),
                "pattern_symmetry": raw.get("pattern_symmetry", ""),
                "numerical_symmetry": raw.get("numerical_symmetry", ""),
            },
        }
        archive = (inspect_root / raw["name"] / f"{raw['name']}.tar.gz") if inspect_root else None
        item["avg_row_length"] = nnz / rows
        item["row_length_variance"] = row_variance(archive) if archive and archive.is_file() else None
        candidates.append(item)

    bins: dict[int, list[dict[str, object]]] = {index: [] for index in range(2, 8)}
    for item in candidates:
        bins[log_bin(int(item["nnz"]))].append(item)
    quotas = {2: 10, 3: 14, 4: 18, 5: 20, 6: 20, 7: 18}
    selected: list[dict[str, object]] = []
    for bucket, quota in quotas.items():
        pool = bins[bucket]
        if len(pool) < quota:
            raise RuntimeError(f"nnz bin 10^{bucket} has only {len(pool)} real matrices, need {quota}")
        pool.sort(key=lambda item: hashlib.sha256(str(item["matrix_id"]).encode()).hexdigest())
        seen: set[tuple[object, ...]] = set()
        while len(selected) < sum(quotas[index] for index in quotas if index < bucket) + quota:
            progressed = False
            for item in pool:
                pattern = item["pattern"]
                key = (
                    pattern["kind"], pattern["is2d3d"], pattern["posdef"],
                    round(float(pattern["pattern_symmetry"] or 0), 1),
                    round(float(item["avg_row_length"]), 1),
                    None if item["row_length_variance"] is None else round(float(item["row_length_variance"]), 1),
                )
                if key in seen:
                    continue
                seen.add(key)
                selected.append(item)
                progressed = True
                if len(selected) >= sum(quotas[index] for index in quotas if index <= bucket):
                    break
            if not progressed:
                break
        if len([item for item in selected if log_bin(int(item["nnz"])) == bucket]) < quota:
            remaining = [item for item in pool if item not in selected]
            selected.extend(remaining[:quota - len([item for item in selected if log_bin(int(item["nnz"])) == bucket])])
    selected.sort(key=lambda item: (int(item["nnz"]), str(item["matrix_id"])))
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} matrices, expected {count}")
    # The downloader uses ``<root>/<name>`` as its stable local directory.
    # Reject a collision here instead of silently overwriting one archive.
    names = [str(item["name"]) for item in selected]
    if len(set(names)) != len(names):
        duplicates = sorted(name for name, occurrences in Counter(names).items() if occurrences > 1)
        raise RuntimeError(f"selected matrices have duplicate names: {', '.join(duplicates)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--inspect-root", type=Path, default=None)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({"http": args.proxy, "https": args.proxy})) if args.proxy else build_opener()
    entries, digest = load_index(opener, args.index)
    selected = select(entries, args.count, args.inspect_root)
    metadata = {
        "dataset_id": "suitesparse_sample_100",
        "selection": {
            "index_url": INDEX_URL,
            "index_sha256": digest,
            "real_only": True,
            "nnz_min": 100,
            "nnz_max": 100_000_000,
            "bin_quotas": {"10^2": 10, "10^3": 14, "10^4": 18, "10^5": 20, "10^6": 20, "10^7": 18},
        },
        "matrices": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    metadata_output = args.metadata_output or args.output.with_suffix(".selection.json")
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "index_sha256": digest, "count": len(selected)}, indent=2))


if __name__ == "__main__":
    main()

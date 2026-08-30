#!/usr/bin/env python3
"""Create a deterministic 100-case Matrix Market smoke corpus.

This is a transport-independent fallback for workers that cannot reach the
SuiteSparse download service.  The generated files keep the public dataset
shape and are suitable for compile, lifecycle, precision, and result-export
regression; production ranking should use the real SuiteSparse manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_matrix(path: Path, index: int) -> dict[str, object]:
    rows = 8 + index % 23
    cols = 8 + (index * 7) % 29
    entries: list[tuple[int, int, int]] = []
    for row in range(rows):
        entries.append((row, (row * 3 + index) % cols, 1 + (row + index) % 11))
        if row % 3 == 0:
            entries.append((row, (row * 5 + 2 * index + 1) % cols, 2 + row % 7))
    unique = {(row, col): value for row, col, value in entries}
    entries = [(row, col, value) for (row, col), value in sorted(unique.items())]
    name = f"synthetic_{index:03d}"
    matrix_dir = path / name
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"{name}.mtx"
    with matrix_path.open("w", encoding="ascii") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write(f"{rows} {cols} {len(entries)}\n")
        for row, col, value in entries:
            stream.write(f"{row + 1} {col + 1} {value}\n")
    return {
        "matrix_id": f"generated/{name}",
        "name": name,
        "group": "generated",
        "rows": rows,
        "cols": cols,
        "nnz": len(entries),
        "source_url": "generated://kernelperf-regression",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    rows = [write_matrix(args.root, index) for index in range(args.count)]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"generated {len(rows)} matrices under {args.root}")


if __name__ == "__main__":
    main()

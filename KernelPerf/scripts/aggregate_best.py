from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(paths: list[Path], *, backend: str, dataset: str, dtype: str, group: str) -> list[dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if (
                row.get("backend_id") != backend
                or row.get("dataset_id") != dataset
                or row.get("dtype") != dtype
                or row.get("candidate_group") != group
                or row.get("selection_role", "candidate") != "candidate"
                or row.get("status") != "pass"
            ):
                continue
            matrix_id = row.get("matrix_id", "")
            if not matrix_id:
                continue
            previous = candidates.get(matrix_id)
            if previous is None or (float(row["solve_ms"]), row.get("configuration_id", "")) < (
                float(previous["solve_ms"]), previous.get("configuration_id", "")
            ):
                candidates[matrix_id] = row
    return list(candidates.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a per-matrix BEST CSV from batched candidate exports")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--candidate-group", required=True)
    args = parser.parse_args()
    paths = sorted(args.input_root.rglob("*.csv"))
    if not paths:
        raise SystemExit(f"no CSV files found under {args.input_root}")
    rows = read_rows(
        paths,
        backend=args.backend,
        dataset=args.dataset,
        dtype=args.dtype,
        group=args.candidate_group,
    )
    if not rows:
        raise SystemExit("no passing candidate rows matched the requested scope")
    selected_from = ",".join(sorted({row.get("configuration_id", "") for row in rows if row.get("configuration_id")}))
    first = rows[0]
    for row in rows:
        row["submission_id"] = f"{args.candidate_group}-best-{args.backend}-{args.dataset}-{args.dtype}"
        row["method_id"] = f"{args.candidate_group.lower()}-best"
        row["method_name"] = f"{args.candidate_group} BEST"
        row["configuration_id"] = "per-matrix-best"
        row["selection_role"] = "best"
        row["selected_from"] = selected_from
        row["base_format"] = "auto"
        row["source_kind"] = "derived"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}: {len(rows)} matrices, selected={selected_from}")


if __name__ == "__main__":
    main()

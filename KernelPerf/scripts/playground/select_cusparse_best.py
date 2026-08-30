from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_COLUMNS = ("backend_id", "operator_id", "dataset_id", "matrix_id")


def select_best(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    fields: list[str] = []
    candidates: dict[tuple[str, ...], dict[str, str]] = {}
    configuration_ids: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            if not fields:
                fields = list(reader.fieldnames)
                for column in ("configuration_id", "candidate_group", "selection_role", "selected_from"):
                    if column not in fields:
                        fields.append(column)
            missing = set(KEY_COLUMNS + ("status", "solve_ms")) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                if row.get("configuration_id"):
                    configuration_ids.add(row["configuration_id"])
                if row.get("status") != "pass":
                    continue
                try:
                    runtime = float(row["solve_ms"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid solve_ms in {path}: {row.get('solve_ms')}") from exc
                if runtime <= 0:
                    continue
                key = tuple(row[column] for column in KEY_COLUMNS)
                previous = candidates.get(key)
                if previous is None or runtime < float(previous["solve_ms"]):
                    candidates[key] = dict(row)

    result = []
    for row in candidates.values():
        row["method_id"] = "CUSPARSE_BEST"
        row["method_name"] = "CUSPARSE_BEST"
        row["configuration_id"] = "per-matrix-best"
        row["candidate_group"] = row.get("candidate_group", "cusparse")
        row["selection_role"] = "best"
        row["selected_from"] = ",".join(sorted(configuration_ids))
        row["base_format"] = "auto"
        row["source_kind"] = "derived"
        result.append(row)
    result.sort(key=lambda row: tuple(row[column] for column in KEY_COLUMNS))
    return fields, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select per-matrix CUSPARSE_BEST rows from cuSPARSE candidate CSVs."
    )
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields, rows = select_best(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"selected {len(rows)} matrices from {len(args.input)} candidate CSV files")


if __name__ == "__main__":
    main()

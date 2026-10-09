from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge checkpointed result-export shards into public result CSVs",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--operator", default="spmv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def timestamp_key(value: str) -> tuple[int, str]:
    text = str(value or "")
    try:
        return (1, datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (0, text)


def row_key(row: dict[str, str]) -> tuple[int, tuple[int, str], str]:
    return (
        1 if row.get("status") == "pass" else 0,
        timestamp_key(row.get("timestamp", "")),
        str(row.get("result_id", "")),
    )


def validate_metadata(rows: list[dict[str, str]], filename: str) -> None:
    identity_columns = (
        "method_id", "method_name", "configuration_id", "candidate_group",
        "selection_role", "selected_from", "base_format", "operator_id", "dtype",
        "backend_id", "hardware", "peak_gflops", "dataset_id", "source_kind",
    )
    for column in identity_columns:
        values = {row.get(column, "") for row in rows if column in row}
        if len(values) > 1:
            raise ValueError(f"inconsistent {column} values for {filename}")


def latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        matrix_id = str(row.get("matrix_id", ""))
        if not matrix_id:
            continue
        previous = selected.get(matrix_id)
        if previous is None or row_key(row) > row_key(previous):
            selected[matrix_id] = row
    return list(selected.values())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    expected_manifest = json.loads(args.matrix_manifest.read_text(encoding="utf-8-sig"))
    matrices = expected_manifest.get("matrices", []) if isinstance(expected_manifest, dict) else expected_manifest
    expected = {str(item["matrix_id"]) for item in matrices}
    index = json.loads(args.index.read_text(encoding="utf-8"))
    entries = {
        (
            str(item.get("method_id", "")),
            str(item.get("backend_id", "")),
            str(item.get("dataset_id", "")),
            str(item.get("dtype", "")),
        ): item
        for item in index.get("submissions", [])
        if str(item.get("operator_id", "")).split(".", 1)[0] == args.operator
    }
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in args.input_root.rglob("*.csv"):
        grouped[path.name].append(path)

    merged = 0
    skipped = 0
    report: list[dict[str, object]] = []
    for filename, paths in sorted(grouped.items()):
        if len(paths) < args.shards:
            skipped += 1
            continue
        fieldnames: list[str] | None = None
        rows: list[dict[str, str]] = []
        for path in paths:
            current_fields, current_rows = read_csv(path)
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"inconsistent columns for {filename}: {path}")
            rows.extend(current_rows)
        validate_metadata(rows, filename)
        selected = latest_rows(rows)
        ids = {str(row.get("matrix_id", "")) for row in selected}
        if ids != expected:
            skipped += 1
            report.append({
                "file": filename,
                "status": "incomplete",
                "shard_files": len(paths),
                "rows": len(selected),
                "missing": len(expected - ids),
                "unexpected": len(ids - expected),
            })
            continue
        first = selected[0]
        key = (
            str(first.get("method_id", "")),
            str(first.get("backend_id", "")),
            str(first.get("dataset_id", "")),
            str(first.get("dtype", "")),
        )
        entry = entries.get(key)
        if entry is None:
            skipped += 1
            report.append({"file": filename, "status": "unindexed", "rows": len(selected)})
            continue
        target = (args.public_root / str(entry["path"])).resolve()
        if not target.is_relative_to(args.public_root.resolve()):
            raise ValueError(f"public result path escapes its root: {entry['path']}")
        if not args.dry_run:
            for row in selected:
                row["submission_id"] = str(entry["submission_id"])
            write_csv(target, fieldnames or [], selected)
            newest = max((row.get("timestamp", "") for row in selected), key=timestamp_key)
            if newest:
                entry["created_at"] = newest
        merged += 1
        report.append({
            "file": filename,
            "status": "merged",
            "shard_files": len(paths),
            "rows": len(selected),
            "pass": sum(row.get("status") == "pass" for row in selected),
            "fail": sum(row.get("status") == "fail" for row in selected),
            "error": sum(row.get("status") == "error" for row in selected),
            "target": str(target),
        })

    if not args.dry_run and merged:
        index["generated_at"] = datetime.now().astimezone().isoformat()
        args.index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"merged": merged, "skipped": skipped, "details": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

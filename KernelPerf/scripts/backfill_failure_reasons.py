from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelperf.benchmark import failure_reason


def filesystem_path(path: Path) -> str:
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return "\\\\?\\" + absolute
    return absolute


def result_key(row: dict[str, Any]) -> tuple[str, ...]:
    timestamp = str(row.get("timestamp", ""))
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            timestamp = parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    return tuple(str(row.get(column, "")) for column in (
        "job_id", "backend_id", "operator_id", "matrix_id", "status",
    )) + (timestamp, str(row.get("rhs_columns", "")), str(row.get("dense_layout", "")))


def load_logs(roots: list[Path]) -> tuple[dict[tuple[str, ...], list[dict[str, Any]]], int]:
    index: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    databases = set()
    for root in roots:
        if root.is_file():
            databases.add(root.resolve())
        else:
            for extension in ("*.sqlite", "*.sqlite3"):
                databases.update(path.resolve() for path in root.rglob(extension))
    for database in sorted(databases):
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            records = connection.execute(
                "SELECT job_id, backend_id, operator_id, matrix_id, kernel_name, "
                "status, timestamp, metadata_json FROM results WHERE status != 'pass'"
            )
            for record in records:
                row = dict(record)
                metadata = json.loads(row.pop("metadata_json"))
                row["rhs_columns"] = metadata.get("rhs_columns", "")
                row["dense_layout"] = metadata.get("dense_layout", "")
                implementation = metadata.get("implementation") or {}
                index[result_key(row)].append({
                    "method_name": row["kernel_name"],
                    "configuration_id": str(implementation.get("configuration_id", "")),
                    "candidate_group": str(implementation.get("candidate_group", "")),
                    "reason": failure_reason(metadata),
                })
        finally:
            connection.close()
    return index, len(databases)


def matches_configuration(row: dict[str, Any], record: dict[str, Any]) -> bool:
    selected = str(row.get("selected_configuration_id") or "")
    if selected and selected != "per-matrix-best":
        configurations = {selected}
    elif row.get("selection_role") == "best":
        configurations = set(str(row.get("selected_from", "")).split(",")) - {""}
    else:
        configurations = {str(row.get("configuration_id", ""))} - {""}
    group = str(row.get("candidate_group", ""))
    if configurations and record.get("configuration_id") and record["configuration_id"] not in configurations:
        return False
    if group and record.get("candidate_group") and group != record["candidate_group"]:
        return False
    if row.get("method_name") and row["method_name"] == record.get("method_name"):
        return True
    return bool(group and group == record.get("candidate_group")
                and record.get("configuration_id") in configurations)


def recover_reason(
    row: dict[str, Any],
    logs: dict[tuple[str, ...], list[dict[str, Any]]],
    candidates: dict[tuple[str, ...], list[dict[str, Any]]],
    *,
    refresh: bool = False,
) -> tuple[str, str]:
    if row.get("status") == "pass":
        return "", "pass"
    metadata = dict(row)
    if refresh:
        metadata.pop("failure_reason", None)
    own_reason = failure_reason(metadata)
    if not own_reason.startswith("historical reason unavailable"):
        return own_reason, "csv_metadata"
    for source, index in (("exact_log", logs), ("candidate_csv", candidates)):
        reasons = {record["reason"] for record in index.get(result_key(row), [])
                   if matches_configuration(row, record)
                   and not record["reason"].startswith("historical reason unavailable")}
        if len(reasons) == 1:
            return reasons.pop(), source
        if len(reasons) > 1:
            return "historical reason unavailable (conflicting retained diagnostics)", "ambiguous"
    return failure_reason({}), "unavailable"


def projection_digest(rows: list[dict[str, str]], columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row.get(column, "") for column in columns],
                                 ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def update_hashes(data_root: Path, *, write: bool) -> int:
    public_root = data_root.parent.resolve()
    updated = 0
    for index_path in data_root.rglob("index.json"):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        changed = False
        for entry in index.get("submissions", []):
            if "csv_sha256" not in entry:
                continue
            target = (public_root / entry["path"]).resolve()
            if not target.is_relative_to(public_root):
                raise ValueError("CSV path escapes the public directory")
            with open(filesystem_path(target), "rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                checksum = digest.hexdigest()
            if checksum != entry["csv_sha256"]:
                entry["csv_sha256"] = checksum
                updated += 1
                changed = True
        if changed and write:
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    return updated


def backfill(
    data_root: Path, database_roots: list[Path], *, write: bool = False, refresh: bool = False,
) -> dict[str, Any]:
    logs, databases = load_logs(database_roots)
    candidates: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    totals: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    unresolved = []
    paths = sorted(data_root.rglob("*.csv"))
    ordered = []
    for path in paths:
        with open(filesystem_path(path), encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"status", "matrix_id"} <= set(reader.fieldnames or []):
                continue
            first = next(reader, {})
            ordered.append((first.get("selection_role") == "best", path))
    for _, path in sorted(ordered):
        with open(filesystem_path(path), encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"malformed CSV: {path.name}")
        original_columns = [column for column in fields if column != "failure_reason"]
        original_digest = projection_digest(rows, original_columns)
        changed = "failure_reason" not in fields
        if changed:
            fields.append("failure_reason")
        unavailable = 0
        for row in rows:
            reason, source = recover_reason(row, logs, candidates, refresh=refresh)
            changed |= row.get("failure_reason", "") != reason
            row["failure_reason"] = reason
            totals[str(row.get("status", "unknown"))] += 1
            sources[source] += 1
            if source in {"unavailable", "ambiguous"}:
                unavailable += 1
            if row.get("status") != "pass" and row.get("selection_role") != "best":
                candidates[result_key(row)].append({
                    "method_name": row.get("method_name", ""),
                    "configuration_id": row.get("configuration_id", ""),
                    "candidate_group": row.get("candidate_group", ""),
                    "reason": reason,
                })
        if unavailable:
            unresolved.append({"path": path.relative_to(data_root).as_posix(),
                               "rows": unavailable})
        if changed and write:
            temporary = path.with_suffix(".csv.failure-reasons.tmp")
            try:
                with open(filesystem_path(temporary), "w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
                with open(filesystem_path(temporary), encoding="utf-8", newline="") as stream:
                    verified_rows = list(csv.DictReader(stream))
                if (len(verified_rows) != len(rows)
                        or projection_digest(verified_rows, original_columns) != original_digest):
                    raise ValueError(f"CSV projection changed: {path.name}")
                os.replace(filesystem_path(temporary), filesystem_path(path))
            finally:
                if os.path.isfile(filesystem_path(temporary)):
                    os.remove(filesystem_path(temporary))
        totals["files"] += 1
        totals["changed_files"] += int(changed)
        if totals["files"] % 250 == 0:
            print(f"processed {totals['files']}/{len(ordered)} CSVs", file=sys.stderr, flush=True)
    return {"write": write, "databases": databases, "totals": dict(totals),
            "reason_sources": dict(sources), "unresolved_files": unresolved,
            "updated_hashes": update_hashes(data_root, write=write)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill CSV failure reasons from exact evaluator log matches")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, action="append", default=[])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = backfill(args.data_root.resolve(), args.database_root,
                      write=args.write, refresh=args.refresh)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "unresolved_files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

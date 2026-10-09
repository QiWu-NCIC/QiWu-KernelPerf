from __future__ import annotations

import csv
import json

from scripts import merge_sharded_results


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_latest_rows_prefers_pass_then_latest_failure():
    rows = [
        {"matrix_id": "a", "status": "pass", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"matrix_id": "a", "status": "fail", "timestamp": "2026-01-02T00:00:00+00:00"},
        {"matrix_id": "b", "status": "error", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"matrix_id": "b", "status": "fail", "timestamp": "2026-01-02T00:00:00+00:00"},
    ]

    selected = {row["matrix_id"]: row for row in merge_sharded_results.latest_rows(rows)}

    assert selected["a"]["status"] == "pass"
    assert selected["b"]["status"] == "fail"


def test_merge_keeps_full_matrix_set_and_normalizes_submission_id(tmp_path, monkeypatch):
    input_root = tmp_path / "exports"
    public_root = tmp_path / "public"
    matrix_manifest = tmp_path / "matrices.json"
    index_path = tmp_path / "index.json"
    matrix_manifest.write_text(json.dumps([{"matrix_id": "a"}, {"matrix_id": "b"}]))
    index_path.write_text(json.dumps({"submissions": [{
        "submission_id": "published-id",
        "method_id": "method",
        "backend_id": "backend",
        "dataset_id": "dataset",
        "dtype": "fp32",
        "operator_id": "spmv.csr.fp32",
        "path": "data/results/spmv/backend/dataset/method.csv",
    }]}))
    fixed = {
        "method_id": "method", "method_name": "Method", "configuration_id": "cfg",
        "candidate_group": "group", "selection_role": "candidate", "selected_from": "",
        "base_format": "csr", "operator_id": "spmv.csr.fp32", "dtype": "fp32",
        "backend_id": "backend", "hardware": "backend", "peak_gflops": "1",
        "dataset_id": "dataset", "source_kind": "source",
    }
    write_rows(input_root / "0" / "method.csv", [
        {**fixed, "submission_id": "job-0", "matrix_id": "a", "status": "pass",
         "timestamp": "2026-01-01T00:00:00+00:00"},
        {**fixed, "submission_id": "job-0", "matrix_id": "b", "status": "fail",
         "timestamp": "2026-01-01T00:00:00+00:00"},
    ])
    write_rows(input_root / "1" / "method.csv", [
        {**fixed, "submission_id": "job-1", "matrix_id": "a", "status": "fail",
         "timestamp": "2026-01-02T00:00:00+00:00"},
        {**fixed, "submission_id": "job-1", "matrix_id": "b", "status": "error",
         "timestamp": "2026-01-02T00:00:00+00:00"},
    ])
    monkeypatch.setattr(merge_sharded_results.sys, "argv", [
        "merge_sharded_results", "--input-root", str(input_root),
        "--public-root", str(public_root), "--matrix-manifest", str(matrix_manifest),
        "--index", str(index_path), "--shards", "2",
    ])

    assert merge_sharded_results.main() == 0

    with (public_root / "data/results/spmv/backend/dataset/method.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["matrix_id"] for row in rows} == {"a", "b"}
    assert {row["submission_id"] for row in rows} == {"published-id"}
    assert {row["matrix_id"]: row["status"] for row in rows} == {"a": "pass", "b": "error"}

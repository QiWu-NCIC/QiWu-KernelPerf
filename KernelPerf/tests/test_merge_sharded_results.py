from __future__ import annotations

import csv
import json

import pytest

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


def test_failure_reason_normalization_handles_old_and_new_headers():
    old_fields = ["matrix_id", "status", "error", "failure_stage"]
    new_fields = old_fields + ["failure_reason"]
    old_rows = [{"matrix_id": "a", "status": "fail", "error": "out of memory",
                 "failure_stage": "setup"}]
    new_rows = [{"matrix_id": "b", "status": "pass", "error": "",
                 "failure_stage": "", "failure_reason": "stale reason"}]
    assert merge_sharded_results.ensure_failure_reason(old_fields, old_rows) == new_fields
    merge_sharded_results.ensure_failure_reason(new_fields, new_rows)
    assert old_rows[0]["failure_reason"] == "setup: out of memory"
    assert new_rows[0]["failure_reason"] == ""


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


def test_spmm_merge_indexes_rhs_columns_and_layout(tmp_path, monkeypatch):
    input_root = tmp_path / "exports"
    public_root = tmp_path / "public"
    matrix_manifest = tmp_path / "matrices.json"
    index_path = tmp_path / "index.json"
    matrix_manifest.write_text(json.dumps([{"matrix_id": "a"}]))
    entries = []
    fixed = {
        "method_id": "method", "method_name": "Method", "configuration_id": "cfg",
        "candidate_group": "group", "selection_role": "candidate", "selected_from": "",
        "base_format": "csr", "operator_id": "spmm.csr.fp32", "dtype": "fp32",
        "backend_id": "backend", "hardware": "backend", "peak_gflops": "1",
        "dataset_id": "dataset", "source_kind": "source", "status": "pass",
        "timestamp": "2026-01-01T00:00:00+00:00", "matrix_id": "a",
        "library_version": "library-1", "solve_only_efficiency_percent": "2",
        "pre_plus_solve_efficiency_percent": "1.5",
        "pre_amortized_efficiency_percent": "1.75",
    }
    for rhs in (2, 4):
        submission_id = f"published-n{rhs}"
        entries.append({
            "submission_id": submission_id, "method_id": "method",
            "backend_id": "backend", "dataset_id": "dataset", "dtype": "fp32",
            "operator_id": "spmm.csr.fp32", "rhs_columns": rhs,
            "dense_layout": "row-major", "path": f"data/results/spmm/n{rhs}.csv",
            "metrics": {},
        })
        write_rows(input_root / str(rhs) / f"method-n{rhs}.csv", [{
            **fixed, "rhs_columns": str(rhs), "dense_layout": "row-major",
            "submission_id": f"job-n{rhs}",
        }])
    index_path.write_text(json.dumps({"submissions": entries}))
    monkeypatch.setattr(merge_sharded_results.sys, "argv", [
        "merge_sharded_results", "--input-root", str(input_root),
        "--public-root", str(public_root), "--matrix-manifest", str(matrix_manifest),
        "--index", str(index_path), "--shards", "1", "--operator", "spmm",
    ])

    assert merge_sharded_results.main() == 0

    for rhs in (2, 4):
        with (public_root / f"data/results/spmm/n{rhs}.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]["submission_id"] == f"published-n{rhs}"
        assert rows[0]["rhs_columns"] == str(rhs)
        entry = next(item for item in json.loads(index_path.read_text())["submissions"]
                     if item["rhs_columns"] == rhs)
        assert entry["total"] == 1
        assert entry["passed"] == 1
        assert entry["failed"] == 0
        assert entry["library_versions"] == ["library-1"]
        assert entry["metrics"]["solve-only"] == 2
        assert len(entry["csv_sha256"]) == 64


def test_best_rows_may_select_different_configurations_per_matrix():
    rows = [
        {"selection_role": "best", "selected_from": "cfg-a", "matrix_id": "a"},
        {"selection_role": "best", "selected_from": "cfg-b", "matrix_id": "b"},
    ]

    merge_sharded_results.validate_metadata(rows, "best.csv")


def test_candidate_metadata_must_keep_selected_from_consistent():
    rows = [
        {"selection_role": "candidate", "selected_from": "cfg-a", "matrix_id": "a"},
        {"selection_role": "candidate", "selected_from": "cfg-b", "matrix_id": "b"},
    ]

    with pytest.raises(ValueError, match="inconsistent selected_from"):
        merge_sharded_results.validate_metadata(rows, "candidate.csv")


def test_partial_merge_appends_only_new_matrix_rows(tmp_path, monkeypatch):
    input_root = tmp_path / "exports"
    public_root = tmp_path / "public"
    matrix_manifest = tmp_path / "matrices.json"
    index_path = tmp_path / "index.json"
    matrix_manifest.write_text(json.dumps([{"matrix_id": "a"}, {"matrix_id": "b"}]))
    relative = "data/results/spmm/n2.csv"
    target = public_root / relative
    fixed = {
        "method_id": "method", "method_name": "Method", "configuration_id": "cfg",
        "candidate_group": "group", "selection_role": "candidate", "selected_from": "cfg",
        "base_format": "csr", "operator_id": "spmm.csr.fp32", "dtype": "fp32",
        "backend_id": "backend", "hardware": "backend", "peak_gflops": "1",
        "dataset_id": "dataset", "source_kind": "source", "status": "pass",
        "timestamp": "2026-01-02T00:00:00+00:00", "rhs_columns": "2",
        "dense_layout": "row-major", "library_version": "library-1",
        "solve_only_efficiency_percent": "2", "pre_plus_solve_efficiency_percent": "1.5",
        "pre_amortized_efficiency_percent": "1.75",
    }
    target.parent.mkdir(parents=True)
    write_rows(target, [{**fixed, "submission_id": "published", "matrix_id": "a",
                         "timestamp": "2026-01-01T00:00:00+00:00"}])
    index_path.write_text(json.dumps({"submissions": [{
        "submission_id": "published", "method_id": "method", "backend_id": "backend",
        "method_name": "Method",
        "dataset_id": "dataset", "dtype": "fp32", "operator_id": "spmm.csr.fp32",
        "rhs_columns": 2, "dense_layout": "row-major", "selection_role": "candidate",
        "path": relative, "metrics": {},
    }]}))
    write_rows(input_root / "shard" / "method.csv", [{
        **fixed, "method_name": "Other Method", "submission_id": "checkpoint", "matrix_id": "b",
    }])
    monkeypatch.setattr(merge_sharded_results.sys, "argv", [
        "merge_sharded_results", "--input-root", str(input_root),
        "--public-root", str(public_root), "--matrix-manifest", str(matrix_manifest),
        "--index", str(index_path), "--shards", "4", "--operator", "spmm",
        "--allow-partial",
    ])

    assert merge_sharded_results.main() == 0

    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["matrix_id"] for row in rows} == {"a", "b"}
    assert all(row["submission_id"] == "published" for row in rows)
    assert all(row["method_name"] == "Method" for row in rows)
    entry = json.loads(index_path.read_text())["submissions"][0]
    assert entry["total"] == 2
    assert entry["passed"] == 2
    assert len(entry["csv_sha256"]) == 64

from __future__ import annotations

import csv
from pathlib import Path

from scripts.aggregate_best import main


def test_aggregate_best_selects_fastest_passing_row(tmp_path, monkeypatch):
    root = tmp_path / "exports"
    root.mkdir()
    path = root / "candidate.csv"
    fields = [
        "submission_id", "method_id", "method_name", "configuration_id", "candidate_group",
        "selection_role", "selected_from", "base_format", "backend_id", "dataset_id", "dtype",
        "matrix_id", "status", "solve_ms", "source_kind",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for config, matrix, elapsed, status in (("a", "m1", 2, "pass"), ("b", "m1", 1, "pass"), ("c", "m2", 1, "error")):
            writer.writerow({
                "submission_id": config, "method_id": config, "method_name": config,
                "configuration_id": config, "candidate_group": "cusparse", "selection_role": "candidate",
                "selected_from": "", "base_format": "csr", "backend_id": "A100-SXM4-80GB",
                "dataset_id": "suite", "dtype": "fp32", "matrix_id": matrix, "status": status,
                "solve_ms": elapsed, "source_kind": "source",
            })
    output = tmp_path / "best.csv"
    monkeypatch.setattr("sys.argv", ["aggregate_best", "--input-root", str(root), "--output", str(output), "--backend", "A100-SXM4-80GB", "--dataset", "suite", "--dtype", "fp32", "--candidate-group", "cusparse"])
    main()
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["matrix_id"] == "m1"
    assert rows[0]["solve_ms"] == "1"
    assert rows[0]["configuration_id"] == "per-matrix-best"
    assert rows[0]["base_format"] == "auto-tuned"
    assert rows[0]["selected_from"] == "a,b,c"


def test_aggregate_best_accepts_public_method_identity(tmp_path, monkeypatch):
    root = tmp_path / "exports"
    root.mkdir()
    path = root / "candidate.csv"
    fields = [
        "submission_id", "method_id", "method_name", "configuration_id", "candidate_group",
        "selection_role", "selected_from", "base_format", "backend_id", "dataset_id", "dtype",
        "matrix_id", "status", "solve_ms", "source_kind",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "submission_id": "scalar", "method_id": "scalar", "method_name": "scalar",
            "configuration_id": "scalar", "candidate_group": "AlphaSparseLib-CSR",
            "selection_role": "candidate", "selected_from": "", "base_format": "csr",
            "backend_id": "gpu", "dataset_id": "suite", "dtype": "fp32",
            "matrix_id": "m1", "status": "pass", "solve_ms": 1, "source_kind": "source",
        })
    output = tmp_path / "best.csv"
    monkeypatch.setattr("sys.argv", [
        "aggregate_best", "--input-root", str(root), "--output", str(output),
        "--backend", "gpu", "--dataset", "suite", "--dtype", "fp32",
        "--candidate-group", "AlphaSparseLib-CSR",
        "--method-id", "AlphaSparseLib-CSR-BEST",
        "--method-name", "AlphaSparseLib CSR BEST",
    ])
    main()
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["method_id"] == "AlphaSparseLib-CSR-BEST"
    assert row["method_name"] == "AlphaSparseLib CSR BEST"

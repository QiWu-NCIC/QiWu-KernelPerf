from __future__ import annotations

import re


RESULT_COLUMNS = [
    "schema_version", "submission_id", "method_id", "method_name", "configuration_id",
    "candidate_group", "selection_role", "selected_from", "base_format",
    "operator_id", "dtype", "backend_id", "hardware", "peak_gflops", "job_id",
    "dataset_id", "matrix_id", "matrix_name", "rows", "cols", "nnz", "status", "operations",
    "preprocess_ms", "solve_ms", "solve_gflops", "solve_only_efficiency_percent",
    "timestamp", "source_kind",
]


def slug(value: str, *, fallback: str = "submission") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or fallback)[:96]

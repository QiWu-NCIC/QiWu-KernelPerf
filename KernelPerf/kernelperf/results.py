from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timezone
from typing import Any


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


def make_submission(
    *,
    job: Any,
    results: list[dict[str, Any]],
    backend: Any,
    operator: Any,
    kernel: Any,
    method_id: str | None = None,
    method_name: str | None = None,
    configuration_id: str | None = None,
    candidate_group: str | None = None,
    selection_role: str | None = None,
    selected_from: str = "",
    source_kind: str | None = None,
    base_format: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    if not results:
        raise ValueError("cannot export an empty result set")
    metadata = kernel.metadata
    method_id = method_id or slug(kernel.name, fallback=job.generator_id)
    method_name = method_name or kernel.name
    configuration_id = configuration_id or str(metadata.get("configuration_id", ""))
    candidate_group = candidate_group or str(metadata.get("candidate_group", ""))
    selection_role = selection_role or str(metadata.get("selection_role", "candidate"))
    source_kind = source_kind or kernel.kind
    identity = configuration_id or method_id
    submission_id = slug(f"{job.job_id}-{backend.backend_id}-{operator.op_id}-{identity}")
    base_format = base_format or str(metadata["base_format"])
    peak = float(backend.peak_gflops_by_dtype.get(operator.dtype, backend.peak_gflops))
    if peak <= 0:
        raise ValueError(f"no positive {operator.dtype} peak configured for {backend.backend_id}")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in results:
        result_metadata = result.get("metadata") or {}
        operations = float(result_metadata.get("operations", 2 * int(result["nnz"])))
        solve_ms = float(result["runtime_ms"])
        solve_gflops = float(result["gflops"])
        efficiency = solve_gflops / peak * 100 if result["status"] == "pass" and solve_gflops > 0 else 0.0
        writer.writerow({
            "schema_version": 2,
            "submission_id": submission_id,
            "method_id": method_id,
            "method_name": method_name,
            "configuration_id": configuration_id,
            "candidate_group": candidate_group,
            "selection_role": selection_role,
            "selected_from": selected_from,
            "base_format": base_format,
            "operator_id": operator.op_id,
            "dtype": operator.dtype,
            "backend_id": backend.backend_id,
            "hardware": backend.name,
            "peak_gflops": peak,
            "job_id": job.job_id,
            "dataset_id": job.dataset_id or "none",
            "matrix_id": result["matrix_id"],
            "matrix_name": result["matrix_name"],
            "rows": result["rows"],
            "cols": result["cols"],
            "nnz": result["nnz"],
            "status": result["status"],
            "operations": operations,
            "preprocess_ms": result["preprocess_ms"],
            "solve_ms": solve_ms,
            "solve_gflops": solve_gflops,
            "solve_only_efficiency_percent": efficiency,
            "timestamp": result["timestamp"],
            "source_kind": source_kind,
        })

    entry = {
        "submission_id": submission_id,
        "method_id": method_id,
        "method_name": method_name,
        "configuration_id": configuration_id,
        "candidate_group": candidate_group,
        "selection_role": selection_role,
        "selected_from": selected_from,
        "base_format": base_format,
        "operator_id": operator.op_id,
        "dtype": operator.dtype,
        "backend_id": backend.backend_id,
        "hardware": backend.name,
        "peak_gflops": peak,
        "dataset_id": job.dataset_id or "none",
        "source_kind": source_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return submission_id, entry, output.getvalue()


from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

from kernelperf.results import slug


SPMM_RESULT_COLUMNS = [
    "schema_version", "submission_id", "method_id", "method_name",
    "configuration_id", "candidate_group", "selection_role", "selection_metric",
    "selected_from", "base_format", "format", "algorithm", "preprocess_policy",
    "operator_id", "dtype", "backend_id", "hardware", "cpu_model",
    "peak_gflops", "job_id", "dataset_id", "matrix_id", "matrix_name",
    "rows", "cols", "nnz", "rhs_columns", "dense_layout", "op_a", "op_b",
    "alpha", "beta", "input_seed", "ranking_scope", "status",
    "validation_status", "failed_elements", "invalid_elements", "operations",
    "preprocess_ms", "solve_ms", "pre_plus_solve_ms", "pre_amortized_ms",
    "solve_gflops", "pre_plus_solve_gflops", "pre_amortized_gflops",
    "solve_only_efficiency_percent", "pre_plus_solve_efficiency_percent",
    "pre_amortized_efficiency_percent", "library_version", "gpu_runtime",
    "runtime_version", "driver_version", "timestamp", "source_kind",
    "selected_configuration_id", "nan_elements", "inf_elements", "error",
    "error_type", "failure_stage",
    "warmup", "iterations", "validation_safety_factor", "public_ranked",
]


def _timing_metrics(operations: float, milliseconds: float, peak: float, passed: bool) -> tuple[float, float]:
    if not passed or milliseconds <= 0:
        return 0.0, 0.0
    gflops = operations / (milliseconds * 1e6)
    return gflops, gflops / peak * 100 if peak > 0 else 0.0


def make_spmm_submission(
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
    selection_metric: str = "",
    selected_from: str = "",
    source_kind: str | None = None,
    base_format: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    if not results:
        raise ValueError("cannot export an empty SpMM result set")
    metadata = kernel.metadata
    method_id = method_id or slug(kernel.name, fallback=job.generator_id)
    method_name = method_name or kernel.name
    configuration_id = configuration_id or str(metadata.get("configuration_id", ""))
    candidate_group = candidate_group or str(metadata.get("candidate_group", ""))
    selection_role = selection_role or str(metadata.get("selection_role", "candidate"))
    source_kind = source_kind or kernel.kind
    identity = configuration_id or method_id
    first_metadata = results[0].get("metadata") or {}
    rhs_columns = int(first_metadata["rhs_columns"])
    dense_layout = str(first_metadata["dense_layout"])
    scopes = {
        (int(item["metadata"]["rhs_columns"]), str(item["metadata"]["dense_layout"]))
        for item in results
    }
    if scopes != {(rhs_columns, dense_layout)}:
        raise ValueError("SpMM CSV cannot mix RHS counts or dense layouts")
    library_versions = sorted({
        str(item.get("metadata", {}).get("library_version", "unknown"))
        for item in results
    } - {"unknown", ""})
    if len(library_versions) > 1:
        raise ValueError("SpMM CSV cannot mix library versions")
    library_version = library_versions[0] if library_versions else "unknown"
    if library_version != "unknown" and library_version not in method_name:
        method_name = f"{method_name} [{library_version}]"
    submission_id = slug(
        f"{job.job_id}-{backend.backend_id}-{operator.op_id}-{identity}-n{rhs_columns}-{selection_metric or 'raw'}"
    )
    base_format = base_format or str(metadata["base_format"])
    peak = float(backend.peak_gflops_by_dtype.get(operator.dtype, backend.peak_gflops))
    if peak <= 0:
        raise ValueError(f"no positive {operator.dtype} peak configured for {backend.backend_id}")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SPMM_RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in results:
        result_metadata = result.get("metadata") or {}
        implementation = result_metadata.get("implementation") or {}
        measurement = result_metadata.get("measurement") or {}
        validation = result_metadata.get("validation") or {}
        validation_status = validation.get("status", validation.get("evaluation_status", "runtime-error"))
        failure_stage = str(result_metadata.get("failure_stage", ""))
        if failure_stage == "compile":
            validation_status = "unavailable"
        error_type = result_metadata.get("error_type", "")
        if not error_type and validation_status in {"precision-error", "invalid-output"}:
            error_type = validation_status
        operations = float(result_metadata.get("operations", 0.0))
        solve_ms = float(result["runtime_ms"])
        preprocess_ms = float(result["preprocess_ms"])
        pre_plus_solve_ms = float(result_metadata.get("pre_plus_solve_ms", preprocess_ms + solve_ms))
        pre_amortized_ms = float(result_metadata.get("pre_amortized_ms", solve_ms))
        passed = result["status"] == "pass"
        solve_gflops, solve_efficiency = _timing_metrics(operations, solve_ms, peak, passed)
        pre_plus_gflops, pre_plus_efficiency = _timing_metrics(
            operations, pre_plus_solve_ms, peak, passed
        )
        pre_amortized_gflops, pre_amortized_efficiency = _timing_metrics(
            operations, pre_amortized_ms, peak, passed
        )
        runtime_version = measurement.get("cuda_runtime_version", measurement.get("hip_runtime_version", ""))
        driver_version = measurement.get("cuda_driver_version", measurement.get("hip_driver_version", ""))
        writer.writerow({
            "schema_version": 3,
            "submission_id": submission_id,
            "method_id": method_id,
            "method_name": method_name,
            "configuration_id": configuration_id,
            "candidate_group": candidate_group,
            "selection_role": selection_role,
            "selection_metric": selection_metric,
            "selected_from": selected_from,
            "base_format": base_format,
            "format": "csr",
            "algorithm": result_metadata.get("algorithm", implementation.get("algorithm", "")),
            "preprocess_policy": implementation.get("preprocess_policy", ""),
            "operator_id": operator.op_id,
            "dtype": operator.dtype,
            "backend_id": backend.backend_id,
            "hardware": backend.name,
            "cpu_model": backend.metadata.get("cpu_model", ""),
            "peak_gflops": peak,
            "job_id": job.job_id,
            "dataset_id": job.dataset_id or "none",
            "matrix_id": result["matrix_id"],
            "matrix_name": result["matrix_name"],
            "rows": result["rows"],
            "cols": result["cols"],
            "nnz": result["nnz"],
            "rhs_columns": result_metadata["rhs_columns"],
            "dense_layout": result_metadata["dense_layout"],
            "op_a": result_metadata["op_a"],
            "op_b": result_metadata["op_b"],
            "alpha": result_metadata["alpha"],
            "beta": result_metadata["beta"],
            "input_seed": result_metadata.get("input_seed", ""),
            "ranking_scope": result_metadata.get("ranking_scope", "main"),
            "status": result["status"],
            "validation_status": validation_status,
            "failed_elements": validation.get("failed_elements", 0),
            "invalid_elements": validation.get("invalid_elements", 0),
            "operations": operations,
            "preprocess_ms": preprocess_ms,
            "solve_ms": solve_ms,
            "pre_plus_solve_ms": pre_plus_solve_ms,
            "pre_amortized_ms": pre_amortized_ms,
            "solve_gflops": solve_gflops,
            "pre_plus_solve_gflops": pre_plus_gflops,
            "pre_amortized_gflops": pre_amortized_gflops,
            "solve_only_efficiency_percent": solve_efficiency,
            "pre_plus_solve_efficiency_percent": pre_plus_efficiency,
            "pre_amortized_efficiency_percent": pre_amortized_efficiency,
            "library_version": result_metadata.get("library_version", "unknown"),
            "gpu_runtime": measurement.get("gpu_runtime", ""),
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "timestamp": result["timestamp"],
            "source_kind": source_kind,
            "selected_configuration_id": implementation.get("configuration_id", configuration_id),
            "nan_elements": measurement.get("nan_elements", 0),
            "inf_elements": measurement.get("inf_elements", 0),
            "error": result_metadata.get("error", ""),
            "error_type": error_type,
            "failure_stage": failure_stage,
            "warmup": measurement.get("warmup", operator.metadata["driver_config"]["warmup"]),
            "iterations": measurement.get("iterations", operator.metadata["driver_config"]["iterations"]),
            "validation_safety_factor": measurement.get("validation_safety_factor", 4.0),
            "public_ranked": str(bool(implementation.get("public_ranked", True))).lower(),
        })

    entry = {
        "submission_id": submission_id,
        "method_id": method_id,
        "method_name": method_name,
        "configuration_id": configuration_id,
        "candidate_group": candidate_group,
        "selection_role": selection_role,
        "selection_metric": selection_metric,
        "selected_from": selected_from,
        "base_format": base_format,
        "format": "csr",
        "operator_id": operator.op_id,
        "dtype": operator.dtype,
        "backend_id": backend.backend_id,
        "hardware": backend.name,
        "peak_gflops": peak,
        "dataset_id": job.dataset_id or "none",
        "rhs_columns": rhs_columns,
        "dense_layout": dense_layout,
        "source_kind": source_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return submission_id, entry, output.getvalue()

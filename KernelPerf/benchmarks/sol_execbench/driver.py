from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kernelperf.benchmark import Benchmark
from kernelperf.models import (
    BenchmarkResult,
    CaseStatus,
    JobRecord,
    KernelArtifact,
    MatrixCase,
    OperatorSpec,
)


def _json_objects(stdout: str) -> list[dict[str, Any]]:
    values: list[Any] = []
    try:
        values.append(json.loads(stdout))
    except json.JSONDecodeError:
        for line in stdout.splitlines():
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _estimate_flops(definition: dict[str, Any], workload: dict[str, Any]) -> float:
    axes = {
        name: float(spec.get("value", 1))
        for name, spec in definition.get("axes", {}).items()
        if spec.get("type") == "const"
    }
    axes.update(
        {key: float(value) for key, value in workload.get("axes", {}).items() if isinstance(value, (int, float))}
    )
    values = [max(value, 1.0) for value in axes.values()]
    op_type = str(definition.get("op_type", "")).lower()
    if op_type in {"rmsnorm", "layernorm", "softmax"}:
        return max(values, default=1.0) * 5.0
    if op_type in {"gemm", "matmul"}:
        m = axes.get("m", axes.get("M", values[0] if values else 1.0))
        n = axes.get("n", axes.get("N", values[1] if len(values) > 1 else 1.0))
        k = axes.get("k", axes.get("K", values[2] if len(values) > 2 else 1.0))
        return 2.0 * m * n * k
    total = 1.0
    for value in values:
        total *= value
    return max(total * 2.0, 1.0)


class SolExecBenchBenchmark(Benchmark):
    def run_case(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
    ) -> list[BenchmarkResult]:
        required = {"problem_dir", "solution_file", "repo_root"}
        missing = sorted(required - set(kernel.metadata))
        if missing:
            raise ValueError(f"SOL-ExecBench artifact is missing metadata: {missing}")
        problem_dir = Path(str(kernel.metadata["problem_dir"]))
        definition = json.loads((problem_dir / "definition.json").read_text())
        workloads = [
            json.loads(line)
            for line in (problem_dir / "workload.jsonl").read_text().splitlines()
            if line.strip()
        ]
        command = [
            str(self.options["python"]),
            "-m",
            str(self.options["module"]),
            str(problem_dir),
            "--solution",
            str(kernel.metadata["solution_file"]),
            "--config",
            str(self.options["config_file"]),
            "--json",
            "--timeout",
            str(self.options["case_timeout_seconds"]),
            "--compile-timeout",
            str(self.options["compile_timeout_seconds"]),
        ]
        completed = backend.run(
            command,
            cwd=str(kernel.metadata["repo_root"]),
            timeout=int(self.options["process_timeout_seconds"]),
        )
        traces = _json_objects(completed.stdout)
        by_uuid = {workload.get("uuid"): workload for workload in workloads}
        matched: set[str] = set()
        reference_hash = hashlib.sha256(
            str(definition.get("reference", "")).encode()
        ).hexdigest()
        info = backend.info()
        results: list[BenchmarkResult] = []

        def make_result(
            workload: dict[str, Any],
            evaluation: dict[str, Any],
            index: int,
            missing_reason: str | None = None,
        ) -> BenchmarkResult:
            evaluation_status = str(evaluation.get("status") or "NO_EVALUATION")
            latency_ms = float((evaluation.get("performance") or {}).get("latency_ms") or 0.0)
            if missing_reason:
                status = CaseStatus.failed
            elif evaluation_status == "PASSED" and latency_ms > 0:
                status = CaseStatus.passed
            elif evaluation_status in {"INCORRECT_SHAPE", "INCORRECT_NUMERICAL", "INCORRECT_DTYPE"}:
                status = CaseStatus.error
            else:
                status = CaseStatus.failed
            workload_id = str(workload.get("uuid") or f"workload-{index}")
            flops = _estimate_flops(definition, workload)
            validation: dict[str, Any] = {
                "method": "standard-program-comparison",
                "evaluation_status": evaluation_status,
                "reference_sha256": reference_hash,
            }
            if evaluation.get("correctness") is not None:
                validation["correctness"] = evaluation["correctness"]
            if missing_reason:
                validation["failure"] = missing_reason
            return BenchmarkResult(
                job_id=job.job_id,
                generator_id=job.generator_id,
                backend_id=info.backend_id,
                backend_kind=info.kind,
                suite=self.benchmark_id,
                operator_id=operator.op_id,
                operator_name=operator.name,
                matrix_id=workload_id,
                matrix_name=workload_id,
                rows=1,
                cols=1,
                nnz=1,
                kernel_name=kernel.name,
                status=status,
                runtime_ms=latency_ms if status == CaseStatus.passed else 0.0,
                gflops=(flops / (latency_ms * 1e6) if status == CaseStatus.passed else 0.0),
                arithmetic_intensity=0.0,
                metadata={
                    "driver": str(self.options["driver_id"]),
                    "validation": validation,
                    "workload": workload,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                },
            )

        for index, trace in enumerate(traces):
            workload = trace.get("workload") or {}
            workload_id = workload.get("uuid")
            if workload_id in by_uuid:
                workload = by_uuid[workload_id]
                matched.add(str(workload_id))
            results.append(make_result(workload, trace.get("evaluation") or {}, index))
        for index, workload in enumerate(workloads, start=len(results)):
            workload_id = str(workload.get("uuid"))
            if workload_id not in matched:
                results.append(
                    make_result(
                        workload,
                        {},
                        index,
                        "benchmark process emitted no trace for this workload",
                    )
                )
        return results

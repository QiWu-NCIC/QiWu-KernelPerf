from __future__ import annotations

import json
import subprocess

from kernelperf.benchmark import benchmark_registry_from_config
from kernelperf.datasets import NO_DATASET_CASE
from kernelperf.models import BackendInfo, CaseStatus, JobRecord, KernelArtifact, OperatorSpec


class TraceBackend:
    backend_id = "local-test"

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def info(self):
        return BackendInfo(
            backend_id=self.backend_id,
            kind="test",
            name="test",
            vendor="test",
            device="test",
            peak_gflops=1,
            memory_bandwidth_gbs=1,
            supported_languages=[],
        )

    def run(self, command, *, cwd, env=None, timeout=None, stdin=None):
        return subprocess.CompletedProcess(command, 1, stdout=self.stdout, stderr="one case failed")


def test_sol_execbench_preserves_pass_error_and_missing_workload_results(tmp_path):
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "definition.json").write_text(
        json.dumps({"name": "test-op", "op_type": "softmax", "reference": "def run(x): return x"})
    )
    workloads = [{"uuid": "pass-case"}, {"uuid": "error-case"}, {"uuid": "missing-case"}]
    (problem_dir / "workload.jsonl").write_text(
        "\n".join(json.dumps(workload) for workload in workloads)
    )
    traces = [
        {
            "workload": workloads[0],
            "evaluation": {
                "status": "PASSED",
                "performance": {"latency_ms": 2.0},
                "correctness": {"max_absolute_error": 0.0},
            },
        },
        {
            "workload": workloads[1],
            "evaluation": {
                "status": "INCORRECT_NUMERICAL",
                "correctness": {"max_absolute_error": 1.0},
            },
        },
    ]
    driver = benchmark_registry_from_config("config/benchmarks.json").get("sol-execbench")
    kernel = KernelArtifact(
        name="candidate",
        entrypoint="solution.json",
        metadata={
            "problem_dir": str(problem_dir),
            "solution_file": str(problem_dir / "solution.json"),
            "repo_root": str(tmp_path),
        },
    )
    job = JobRecord(
        generator_id="test",
        backends=["local-test"],
        suites=["sol-execbench"],
        kernels=[kernel],
    )

    results = driver.run_case(
        TraceBackend("\n".join(json.dumps(trace) for trace in traces)),
        job,
        OperatorSpec(op_id="test-op", name="Test", suite="sol-execbench"),
        NO_DATASET_CASE,
        kernel,
    )

    assert {result.matrix_id: result.status for result in results} == {
        "pass-case": CaseStatus.passed,
        "error-case": CaseStatus.error,
        "missing-case": CaseStatus.failed,
    }
    assert all(
        result.metadata["validation"]["method"] == "standard-program-comparison"
        for result in results
    )

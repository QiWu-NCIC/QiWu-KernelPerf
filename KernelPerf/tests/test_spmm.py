from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.spmm.driver import validate_kernel
from benchmarks.spmm.results import make_spmm_submission
from scripts.run_spmm_campaign import kernel_key, row_key
from kernelperf.backends import LocalBackend
from kernelperf.benchmark import benchmark_registry_from_config
from kernelperf.exports import LocalResultExporter
from kernelperf.models import BenchmarkResult, CaseStatus, JobRecord, JobStatus, KernelArtifact, MatrixCase
from kernelperf.runtime import create_runtime
from kernelperf.submissions import load_submission_artifacts


def driver():
    return benchmark_registry_from_config("config/benchmarks.json").get("spmm")


def kernel():
    return KernelArtifact(
        name="incorrect-fixture", language="cuda", entrypoint="qiwu_spmm_plugin",
        source=Path("tests/fixtures/spmm_incorrect.cu").read_text(),
        metadata={"operator_id": "spmm.csr.fp32", "base_format": "csr"},
    )


@pytest.mark.parametrize("name,language,count", [
    ("alphasparse", "cuda", 5), ("alphasparse", "hip", 5),
    ("cusparse", "cuda", 4), ("rocsparse_dtk2604", "hip", 4),
])
def test_p0_manifests_and_standalone_packages(name, language, count):
    artifacts = load_submission_artifacts(Path("submissions/spmm") / name, language=language)
    assert len(artifacts) == count
    benchmark = driver()
    for artifact in artifacts:
        assert artifact.entrypoint == "qiwu_spmm_plugin"
        validate_kernel(artifact, benchmark.options)
    package = benchmark.source_package(artifacts[0])
    files = {item["path"]: item["content"] for item in package["files"]}
    assert "examples/standalone.cu" in files
    assert "qiwu_spmm_preprocess" in files["examples/standalone.cu"]
    assert "qiwu/spmm_plugin.cuh" in files["examples/standalone.cu"]
    assert package["plugin"]["language"] == language
    assert f"LANGUAGES CXX {language.upper()}" in files["CMakeLists.txt"]
    assert "SpMV" not in files["README-QIWU-PLUGIN.md"]
    if name == "alphasparse":
        assert any(path.startswith(f"upstream/{language}/kernel/level3/") for path in files)
        assert not any(path.startswith(f"upstream/{'hip' if language == 'cuda' else 'cuda'}/") for path in files)
        if language == "cuda":
            adapter = files["adapter.cu"]
            upstream = files["upstream/cuda/kernel/level3/alphasparse_spmm.cu"]
            assert "#define QIWU_SPMM_REAL_TYPES_ONLY 1" in adapter
            assert "#if !defined(QIWU_SPMM_REAL_TYPES_ONLY)" in upstream


def test_protocol_and_independent_contract():
    benchmark = driver()
    for operator in benchmark.operators():
        protocol = operator.metadata["driver_config"]
        assert protocol["rhs_columns"] == [2, 4, 8, 16, 32, 64, 128]
        assert protocol["sanity_rhs_columns"] == [1]
        assert protocol["warmup"] == 5
        assert protocol["iterations"] == 20
        assert protocol["validation_safety_factor"] == 4
        assert protocol["input_seed"] == 20260922
        assert protocol["dense_layout"] == "row-major"
    source = benchmark._source(kernel(), benchmark.operators()[0])
    assert "standard_spmm" in source
    assert "long double" in source
    assert "QiwuSpmv" not in source
    assert "qiwu_spmm_plugin" not in Path("include/qiwu/spmv_plugin.cuh").read_text()


def test_campaign_checkpoint_key_keeps_configurations_separate():
    base = {
        "matrix_id": "group/rect",
        "metadata": {
            "rhs_columns": 2,
            "implementation": {"candidate_group": "AlphaSparse-SpMM-CSR", "configuration_id": "csr-alg1"},
        },
    }
    other = {
        **base,
        "metadata": {
            "rhs_columns": 2,
            "implementation": {"candidate_group": "cuSPARSE-SpMM-CSR", "configuration_id": "csr-alg1"},
        },
    }
    assert row_key({"kernel_name": "same method name", **base}) != row_key({"kernel_name": "same method name", **other})
    artifact = KernelArtifact(
        name="same method name", language="cuda", entrypoint="qiwu_spmm_plugin",
        source="", metadata=base["metadata"]["implementation"],
    )
    assert (*kernel_key(artifact), "group/rect", 2) == row_key({"kernel_name": artifact.name, **base})


class FakeBackend(LocalBackend):
    def __init__(self, valid=True, error=False, fail_rhs=None):
        super().__init__({
            "worker_id": "test", "backend_id": "test", "endpoint": "local",
            "transport": "local", "kind": "gpu", "peak_gflops": 100,
        })
        self.valid = valid
        self.error = error
        self.fail_rhs = fail_rhs
        self.calls = 0

    def path_exists(self, path, *, executable=False):
        return True

    def run(self, command, *, cwd, env=None, timeout=None, stdin=None):
        self.calls += 1
        if self.error:
            return subprocess.CompletedProcess(command, 1, '{"status":"error","stage":"setup","error":"matrix dimensions mismatch"}', "")
        payloads = []
        for columns in map(int, env["KERNELPERF_RHS_COLUMNS"].split(",")):
            if columns == self.fail_rhs:
                payloads.append({
                    "status": "error", "stage": "solve",
                    "error": "intentional RHS failure", "rhs_columns": columns,
                })
                continue
            payloads.append({
                "preprocess_ms": 2.0, "runtime_ms": 0.5,
                "pre_plus_solve_ms": 2.5, "pre_amortized_ms": 0.6,
                "operations": 2 * 5 * columns, "valid": self.valid,
                "metadata": {"actual_nnz": 5, "rhs_columns": columns,
                             "failed_elements": 0 if self.valid else 3,
                             "validation_status": "pass" if self.valid else "precision-error",
                             "library_version": "test 1.2.3"},
            })
        return subprocess.CompletedProcess(
            command, 0, "\n".join(json.dumps(payload) for payload in payloads), ""
        )


def test_driver_scores_each_rhs_and_retains_failures(tmp_path):
    benchmark = driver()
    artifact = kernel()
    job = JobRecord(generator_id="test", backends=["test"], suites=["spmm"], kernels=[artifact])
    matrix = MatrixCase(matrix_id="group/rect", name="rect", rows=3, cols=4, nnz=5, local_path=str(tmp_path))
    operator = benchmark.operators()[0]
    for backend, status in [(FakeBackend(), CaseStatus.passed), (FakeBackend(False), CaseStatus.error), (FakeBackend(error=True), CaseStatus.failed)]:
        results = benchmark.run_case(backend, job, operator, matrix, artifact)
        assert backend.calls == 1
        assert len(results) == 8
        assert all(result.status == status for result in results)
        assert [result.metadata["rhs_columns"] for result in results] == [1, 2, 4, 8, 16, 32, 64, 128]
        assert results[0].metadata["ranking_scope"] == "sanity"
        if status == CaseStatus.failed:
            assert results[1].metadata["validation"]["evaluation_status"] == "dimension-error"
            _, _, csv_content = make_spmm_submission(
                job=job,
                results=[results[1].model_dump()],
                backend=backend.info(),
                operator=operator,
                kernel=artifact,
            )
            row = next(csv.DictReader(io.StringIO(csv_content)))
            assert row["error_type"] == "BenchmarkDriverError"
            assert row["failure_stage"] == "setup"
        if status == CaseStatus.error:
            assert all(result.metadata["validation"]["status"] == "precision-error" for result in results)
            assert all(result.metadata["validation"]["failed_elements"] == 3 for result in results)
        if status == CaseStatus.passed:
            assert results[2].gflops == pytest.approx(2 * 5 * 4 / 0.5 / 1e6)
        else:
            assert all(result.gflops == 0 for result in results)


def test_driver_retains_other_rhs_when_one_runtime_fails(tmp_path):
    benchmark = driver()
    artifact = kernel()
    job = JobRecord(generator_id="test", backends=["test"], suites=["spmm"], kernels=[artifact])
    matrix = MatrixCase(matrix_id="group/rect", name="rect", rows=3, cols=4, nnz=5, local_path=str(tmp_path))
    backend = FakeBackend(fail_rhs=8)
    results = benchmark.run_case(backend, job, benchmark.operators()[0], matrix, artifact)
    assert backend.calls == 1
    assert [result.metadata["rhs_columns"] for result in results] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert [result.status for result in results] == [
        CaseStatus.passed, CaseStatus.passed, CaseStatus.passed, CaseStatus.failed,
        CaseStatus.passed, CaseStatus.passed, CaseStatus.passed, CaseStatus.passed,
    ]


def test_best_scopes_and_metric_selection(tmp_path):
    runtime = create_runtime(database_path=tmp_path / "test.sqlite", result_exports_path=tmp_path / "exports")
    benchmark = runtime.benchmarks.get("spmm")
    operator = benchmark.operators()[0]
    backend = runtime.backends.backends()[0].info()
    artifacts = []
    for configuration in ("a", "b", "default"):
        artifact = kernel()
        artifact.name = configuration
        artifact.metadata.update(configuration_id=configuration, candidate_group="test-group", public_ranked=configuration != "default")
        artifacts.append(artifact)
    job = JobRecord(generator_id="test", backends=[backend.backend_id], suites=["spmm"], dataset_id="test", operator_ids=[operator.op_id], kernels=artifacts, status=JobStatus.succeeded)
    for columns in (1, 2, 8):
        for artifact in artifacts:
            solve = {"a": 1.0, "b": 2.0, "default": 0.1}[artifact.name]
            preprocess = 10.0 if artifact.name == "a" else 0.0
            result = BenchmarkResult(
                job_id=job.job_id, generator_id="test", backend_id=backend.backend_id,
                backend_kind=backend.kind, suite="spmm", operator_id=operator.op_id,
                operator_name=operator.name, matrix_id="rect", matrix_name="rect",
                rows=3, cols=4, nnz=5, kernel_name=artifact.name,
                runtime_ms=solve, preprocess_ms=preprocess, gflops=0,
                arithmetic_intensity=0, metadata={
                    "implementation": artifact.metadata, "dtype": "fp32", "rhs_columns": columns,
                    "dense_layout": "row-major", "op_a": "N", "op_b": "N", "alpha": 1, "beta": 0,
                    "operations": 2 * 5 * columns, "ranking_scope": "sanity" if columns == 1 else "main",
                    "pre_plus_solve_ms": preprocess + solve, "pre_amortized_ms": preprocess / 20 + solve,
                    "library_version": "test 1.2.3", "validation": {"status": "pass"},
                },
            )
            runtime.db.insert_result(result)
    exporter = LocalResultExporter(tmp_path / "exports", runtime.db, runtime.backends, runtime.benchmarks)
    paths = exporter.export_job(job)
    assert len(paths) == 15
    best_files = [Path(path) for path in paths if "-best-" in Path(path).name]
    assert len(best_files) == 6
    for path in best_files:
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        assert len(rows) == 1
        row = rows[0]
        assert int(row["rhs_columns"]) in (2, 8)
        assert row["selected_from"] == "a,b"
        assert row["selected_configuration_id"] == ("b" if row["selection_metric"] == "pre-plus-solve" else "a")
        assert "test 1.2.3" in row["method_name"]
    rows = runtime.db.query_results(job_ids=[job.job_id], suites=["spmm"])
    with pytest.raises(ValueError, match="mix RHS"):
        make_spmm_submission(job=job, results=rows, backend=backend, operator=operator, kernel=artifacts[0])


def test_latest_spmm_failure_is_exported_and_excluded_from_best(tmp_path):
    runtime = create_runtime(database_path=tmp_path / "test.sqlite", result_exports_path=tmp_path / "exports")
    benchmark = runtime.benchmarks.get("spmm")
    operator = benchmark.operators()[0]
    backend = runtime.backends.backends()[0].info()
    artifacts = []
    for configuration in ("a", "b"):
        artifact = kernel()
        artifact.name = configuration
        artifact.metadata.update(configuration_id=configuration, candidate_group="test-group")
        artifacts.append(artifact)
    job = JobRecord(
        generator_id="test", backends=[backend.backend_id], suites=["spmm"],
        dataset_id="test", operator_ids=[operator.op_id], kernels=artifacts,
        status=JobStatus.succeeded,
    )

    def insert(artifact, status, runtime_ms, timestamp):
        passed = status == CaseStatus.passed
        runtime.db.insert_result(BenchmarkResult(
            job_id=job.job_id, generator_id="test", backend_id=backend.backend_id,
            backend_kind=backend.kind, suite="spmm", operator_id=operator.op_id,
            operator_name=operator.name, matrix_id="rect", matrix_name="rect",
            rows=3, cols=4, nnz=5, kernel_name=artifact.name, status=status,
            runtime_ms=runtime_ms, preprocess_ms=0.0, gflops=0.0,
            arithmetic_intensity=0.0, timestamp=timestamp,
            metadata={
                "implementation": artifact.metadata, "dtype": "fp32",
                "rhs_columns": 2, "dense_layout": "row-major", "op_a": "N",
                "op_b": "N", "alpha": 1.0, "beta": 0.0,
                "operations": 20, "ranking_scope": "main",
                "pre_plus_solve_ms": runtime_ms, "pre_amortized_ms": runtime_ms,
                "library_version": "test 1.2.3",
                "validation": {"status": "pass" if passed else "precision-error"},
                "error_type": "" if passed else "ValidationError",
                "failure_stage": "" if passed else "validation",
                "error": "" if passed else "incorrect output",
            },
        ))

    insert(artifacts[0], CaseStatus.passed, 0.1, "2026-09-23T00:00:00+00:00")
    insert(artifacts[0], CaseStatus.failed, 0.0, "2026-09-23T00:00:01+00:00")
    insert(artifacts[1], CaseStatus.passed, 1.0, "2026-09-23T00:00:00+00:00")

    paths = LocalResultExporter(
        tmp_path / "exports", runtime.db, runtime.backends, runtime.benchmarks
    ).export_job(job)
    candidate_path = next(Path(path) for path in paths if Path(path).name.startswith("a-"))
    with candidate_path.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    assert len(candidate_rows) == 1
    assert candidate_rows[0]["status"] == "fail"
    assert candidate_rows[0]["failure_stage"] == "validation"

    best_paths = [Path(path) for path in paths if "-best-" in Path(path).name]
    assert len(best_paths) == 3
    for path in best_paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["selected_configuration_id"] == "b"

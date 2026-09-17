from __future__ import annotations

import csv
from pathlib import Path

from kernelperf.exports import LocalResultExporter
from kernelperf.models import BenchmarkResult, JobRecord, JobStatus, KernelArtifact
from kernelperf.runtime import create_runtime


def test_successful_spmv_job_is_exported_as_canonical_csv(tmp_path):
    runtime_obj = create_runtime(database_path=tmp_path / "perf.sqlite", result_exports_path=tmp_path / "exports")
    backend = runtime_obj.backends.backends()[0].info()
    operator = runtime_obj.benchmarks.get("spmv").operators()[0]
    kernel = KernelArtifact(
        name="SELL-C32",
        source="candidate",
        metadata={"operator_id": operator.op_id, "base_format": "sell"},
    )
    job = JobRecord(
        job_id="automatic-export",
        generator_id="SELL-C32",
        backends=[backend.backend_id],
        suites=["spmv"],
        dataset_id="suitesparse_sample_100",
        operator_ids=[operator.op_id],
        kernels=[kernel],
        status=JobStatus.succeeded,
    )
    runtime_obj.db.insert_result(BenchmarkResult(
        job_id=job.job_id,
        generator_id=job.generator_id,
        backend_id=backend.backend_id,
        backend_kind=backend.kind,
        suite="spmv",
        operator_id=operator.op_id,
        operator_name=operator.name,
        matrix_id="group/matrix",
        matrix_name="matrix",
        rows=10,
        cols=10,
        nnz=20,
        kernel_name=kernel.name,
        preprocess_ms=2.0,
        runtime_ms=0.5,
        gflops=0.00008,
        arithmetic_intensity=0.0,
        metadata={"operations": 40, "dtype": operator.dtype},
    ))
    exporter = LocalResultExporter(
        tmp_path / "exports",
        runtime_obj.db,
        runtime_obj.backends,
        runtime_obj.benchmarks,
    )

    paths = exporter.export_job(job)

    assert len(paths) == 1
    result_path = Path(paths[0])
    assert result_path.parent == tmp_path / "exports" / "spmv" / backend.backend_id / "suitesparse_sample_100"
    with result_path.open(newline="", encoding="utf-8") as result_file:
        rows = list(csv.DictReader(result_file))
    assert rows[0]["job_id"] == job.job_id
    assert rows[0]["schema_version"] == "2"
    assert "solve_only_efficiency_percent" in rows[0]
    assert "efficiency_percent" not in rows[0]
    assert rows[0]["base_format"] == "sell"
    assert rows[0]["matrix_id"] == "group/matrix"
    assert not list(result_path.parent.glob("*.tmp"))
    assert exporter.export_job(job) == paths


def test_failed_job_does_not_export_csv(tmp_path):
    runtime = create_runtime(database_path=tmp_path / "perf.sqlite", result_exports_path=tmp_path / "exports")
    exporter = LocalResultExporter(
        tmp_path / "exports",
        runtime.db,
        runtime.backends,
        runtime.benchmarks,
    )
    job = JobRecord(
        generator_id="failed",
        backends=[runtime.backends.backends()[0].backend_id],
        suites=["spmv"],
        kernels=[KernelArtifact(name="candidate")],
        status=JobStatus.failed,
    )

    assert exporter.export_job(job) == []
    assert not (tmp_path / "exports").exists()


def test_datasets_have_independent_export_paths(tmp_path):
    runtime = create_runtime(database_path=tmp_path / "perf.sqlite", result_exports_path=tmp_path / "exports")
    backend = runtime.backends.backends()[0].info()
    operator = runtime.benchmarks.get("spmv").operators()[0]
    kernel = KernelArtifact(
        name="candidate", source="candidate",
        metadata={"operator_id": operator.op_id, "base_format": "csr"},
    )
    exporter = LocalResultExporter(
        tmp_path / "exports", runtime.db, runtime.backends, runtime.benchmarks
    )
    paths = []
    for dataset_id, job_id in (("suite-a", "job-a"), ("suite-b", "job-b")):
        job = JobRecord(
            job_id=job_id,
            generator_id="candidate",
            backends=[backend.backend_id],
            suites=["spmv"],
            dataset_id=dataset_id,
            operator_ids=[operator.op_id],
            kernels=[kernel],
            status=JobStatus.succeeded,
        )
        runtime.db.insert_result(BenchmarkResult(
            job_id=job_id,
            generator_id="candidate",
            backend_id=backend.backend_id,
            backend_kind=backend.kind,
            suite="spmv",
            operator_id=operator.op_id,
            operator_name=operator.name,
            matrix_id="group/matrix",
            matrix_name="matrix",
            rows=2,
            cols=2,
            nnz=2,
            kernel_name=kernel.name,
            runtime_ms=1.0,
            gflops=0.004,
            arithmetic_intensity=0.0,
            metadata={"operations": 4, "dtype": operator.dtype},
        ))
        paths.extend(exporter.export_job(job))

    assert len(paths) == 2
    assert {Path(path).parent.name for path in paths} == {"suite-a", "suite-b"}
    assert all(Path(path).is_file() for path in paths)


def test_multi_configuration_job_exports_candidates_and_per_matrix_best(tmp_path):
    runtime_obj = create_runtime(database_path=tmp_path / "perf.sqlite", result_exports_path=tmp_path / "exports")
    backend = runtime_obj.backends.backends()[0].info()
    operator = runtime_obj.benchmarks.get("spmv").operators()[0]
    kernels = [
        KernelArtifact(
            name="cfg-a", source="candidate",
            metadata={
                "operator_id": operator.op_id, "base_format": "csr",
                "configuration_id": "a", "candidate_group": "alpha",
            },
        ),
        KernelArtifact(
            name="cfg-b", source="candidate",
            metadata={
                "operator_id": operator.op_id, "base_format": "csr",
                "configuration_id": "b", "candidate_group": "alpha",
            },
        ),
    ]
    job = JobRecord(
        job_id="multi-config",
        generator_id="alpha",
        backends=[backend.backend_id],
        suites=["spmv"],
        dataset_id="suitesparse_sample_100",
        operator_ids=[operator.op_id],
        kernels=kernels,
        status=JobStatus.succeeded,
    )
    for kernel, timings in zip(kernels, ([1.0, 3.0], [2.0, 0.5])):
        for matrix_id, elapsed in zip(("m1", "m2"), timings):
            runtime_obj.db.insert_result(BenchmarkResult(
                job_id=job.job_id,
                generator_id=job.generator_id,
                backend_id=backend.backend_id,
                backend_kind=backend.kind,
                suite="spmv",
                operator_id=operator.op_id,
                operator_name=operator.name,
                matrix_id=matrix_id,
                matrix_name=matrix_id,
                rows=2,
                cols=2,
                nnz=4,
                kernel_name=kernel.name,
                runtime_ms=elapsed,
                gflops=8.0 / elapsed,
                arithmetic_intensity=0.0,
                metadata={
                    "operations": 8,
                    "dtype": operator.dtype,
                    "implementation": {
                        "base_format": "csr",
                        "configuration_id": kernel.metadata["configuration_id"],
                        "candidate_group": "alpha",
                        "selection_role": "candidate",
                    },
                },
            ))
    exporter = LocalResultExporter(
        tmp_path / "exports", runtime_obj.db, runtime_obj.backends, runtime_obj.benchmarks
    )
    paths = exporter.export_job(job)
    assert len(paths) == 3
    best = next(Path(path) for path in paths if "best" in Path(path).name)
    with best.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["solve_ms"] for row in rows} == {"1.0", "0.5"}
    assert all(row["configuration_id"] == "per-matrix-best" for row in rows)
    assert all(row["selection_role"] == "best" for row in rows)
    assert all(row["base_format"] == "manual-selection" for row in rows)
    assert all(row["source_kind"] == "derived" for row in rows)


def test_multiple_candidate_groups_export_independent_best_results(tmp_path):
    runtime = create_runtime(
        database_path=tmp_path / "perf.sqlite",
        result_exports_path=tmp_path / "exports",
    )
    backend = runtime.backends.backends()[0].info()
    operator = runtime.benchmarks.get("spmv").operators()[0]
    kernels = [
        KernelArtifact(
            name=f"{group}-{configuration}",
            source="candidate",
            metadata={
                "operator_id": operator.op_id,
                "base_format": "csr",
                "configuration_id": configuration,
                "candidate_group": group,
            },
        )
        for group in ("alpha", "beta")
        for configuration in ("a", "b")
    ]
    job = JobRecord(
        job_id="multiple-groups",
        generator_id="sweep",
        backends=[backend.backend_id],
        suites=["spmv"],
        dataset_id="suite",
        operator_ids=[operator.op_id],
        kernels=kernels,
        status=JobStatus.succeeded,
    )
    for index, kernel in enumerate(kernels):
        runtime.db.insert_result(BenchmarkResult(
            job_id=job.job_id,
            generator_id=job.generator_id,
            backend_id=backend.backend_id,
            backend_kind=backend.kind,
            suite="spmv",
            operator_id=operator.op_id,
            operator_name=operator.name,
            matrix_id="m1",
            matrix_name="m1",
            rows=2,
            cols=2,
            nnz=4,
            kernel_name=kernel.name,
            runtime_ms=float(index + 1),
            gflops=8.0 / (index + 1),
            arithmetic_intensity=0.0,
            metadata={
                "operations": 8,
                "dtype": operator.dtype,
                "implementation": {
                    "base_format": "csr",
                    "configuration_id": kernel.metadata["configuration_id"],
                    "candidate_group": kernel.metadata["candidate_group"],
                    "selection_role": "candidate",
                },
            },
        ))

    paths = LocalResultExporter(
        tmp_path / "exports", runtime.db, runtime.backends, runtime.benchmarks
    ).export_job(job)

    assert len(paths) == 6
    best_paths = sorted(Path(path).name for path in paths if "best" in Path(path).name)
    assert best_paths == [
        f"alpha-best-{backend.backend_id}-suite-{operator.dtype}.csv",
        f"beta-best-{backend.backend_id}-suite-{operator.dtype}.csv",
    ]

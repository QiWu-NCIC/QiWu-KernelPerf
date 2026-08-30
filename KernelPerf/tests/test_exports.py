from __future__ import annotations

import csv
from pathlib import Path

from kernelperf.app import create_app
from kernelperf.exports import LocalResultExporter
from kernelperf.models import BenchmarkResult, JobRecord, JobStatus, KernelArtifact


def test_successful_spmv_job_is_exported_as_canonical_csv(tmp_path):
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    backend = app.state.backends.backends()[0].info()
    operator = app.state.benchmarks.get("spmv").operators()[0]
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
    app.state.db.insert_result(BenchmarkResult(
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
        app.state.db,
        app.state.backends,
        app.state.benchmarks,
    )

    paths = exporter.export_job(job)

    assert len(paths) == 1
    result_path = Path(paths[0])
    assert result_path.parent == tmp_path / "exports" / "spmv" / backend.backend_id
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
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    exporter = LocalResultExporter(
        tmp_path / "exports",
        app.state.db,
        app.state.backends,
        app.state.benchmarks,
    )
    job = JobRecord(
        generator_id="failed",
        backends=[app.state.backends.backends()[0].backend_id],
        suites=["spmv"],
        kernels=[KernelArtifact(name="candidate")],
        status=JobStatus.failed,
    )

    assert exporter.export_job(job) == []
    assert not (tmp_path / "exports").exists()


def test_multi_configuration_job_exports_candidates_and_per_matrix_best(tmp_path):
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    backend = app.state.backends.backends()[0].info()
    operator = app.state.benchmarks.get("spmv").operators()[0]
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
        for matrix_id, runtime in zip(("m1", "m2"), timings):
            app.state.db.insert_result(BenchmarkResult(
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
                runtime_ms=runtime,
                gflops=8.0 / runtime,
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
        tmp_path / "exports", app.state.db, app.state.backends, app.state.benchmarks
    )
    paths = exporter.export_job(job)
    assert len(paths) == 3
    best = next(Path(path) for path in paths if "best" in Path(path).name)
    with best.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["solve_ms"] for row in rows} == {"1.0", "0.5"}
    assert all(row["configuration_id"] == "per-matrix-best" for row in rows)
    assert all(row["selection_role"] == "best" for row in rows)
    assert all(row["source_kind"] == "derived" for row in rows)

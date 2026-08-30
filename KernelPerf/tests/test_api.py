from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from kernelperf.app import create_app
from kernelperf.backends import backend_registry_from_config
from kernelperf.database import PerfDatabase
from kernelperf.models import (
    BenchmarkResult,
    JobRecord,
    JobStatus,
    KernelArtifact,
    WorkerNode,
)


def test_worker_configuration_registers_transports_without_code_branches():
    registry = backend_registry_from_config("config/workers.json")
    backends = {backend.backend_id: backend for backend in registry.backends()}
    configured = json.loads(Path("config/workers.json").read_text())
    expected = {item["backend_id"]: item["transport"] for item in configured}

    assert set(backends) == set(expected)
    assert {
        backend_id: backend.info().metadata["transport"]
        for backend_id, backend in backends.items()
    } == expected


def test_unknown_worker_is_rejected_before_queueing(tmp_path):
    benchmark = next(
        item for item in json.loads(Path("config/benchmarks.json").read_text())
        if item.get("default")
    )
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            json={
                "generator_id": "client",
                "backends": ["not-configured"],
                "suites": [benchmark["benchmark_id"]],
                "dataset_id": benchmark["default_dataset_id"],
                "kernels": [{"name": "candidate", "source": "invalid"}],
            },
        )

    assert response.status_code == 400
    assert "Unknown backend" in response.json()["detail"]


def test_frontend_adapter_serves_uncached_assets(tmp_path):
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    with TestClient(app) as client:
        index = client.get("/")
        assert 'href="/static/styles.css"' in index.text
        assert 'src="/static/app.js"' in index.text
        submit = client.get("/submit")
        assert 'src="/static/submit.js"' in submit.text
        assert '<select id="baseFormat" required>' in submit.text
        for label in ["CSR", "COO", "ELL", "SELL", "HYB", "BSR", "DIA", "Auto-tuned", "Unmarked"]:
            assert f">{label}</option>" in submit.text
        app_script = client.get("/static/app.js")
        assert 'fillText("GFLOP/s"' in app_script.text
        assert "Number(row.gflops)" in app_script.text
        for path in ["/", "/submit", "/static/app.js", "/static/styles.css", "/static/qiwu-logo.png"]:
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"].startswith("no-store")


def test_worker_registry_prunes_stale_rows(tmp_path):
    db_path = tmp_path / "perf.sqlite"
    db = PerfDatabase(db_path)
    db.upsert_worker(WorkerNode(worker_id="retired", backend_id="retired"))
    app = create_app(str(db_path), str(tmp_path / "contest.sqlite"))

    with TestClient(app) as client:
        workers = client.get("/api/v1/workers").json()

    expected = {
        item["worker_id"]
        for item in json.loads(Path("config/workers.json").read_text())
    }
    assert {worker["worker_id"] for worker in workers} == expected
    assert {worker["worker_id"] for worker in db.list_workers()} == expected


def test_job_logs_are_complete_and_plot_logs_are_scoped(tmp_path):
    db = PerfDatabase(tmp_path / "perf.sqlite")
    for index in range(150):
        db.append_job_log("job-1", f"timestamp-{index:03d}", f"line {index}", "worker")
    assert len(db.list_job_logs("job-1")) == 150

    app = create_app(str(tmp_path / "api.sqlite"), str(tmp_path / "contest.sqlite"))
    common = {
        "job_id": "job-shared",
        "generator_id": "generator",
        "backend_kind": "gpu",
        "suite": "benchmark",
        "operator_id": "operator",
        "operator_name": "Operator",
        "matrix_id": "case",
        "matrix_name": "case",
        "rows": 1,
        "cols": 1,
        "nnz": 1,
        "kernel_name": "candidate",
        "preprocess_ms": 3.0,
        "runtime_ms": 1.0,
        "gflops": 2.0,
        "arithmetic_intensity": 0.0,
    }
    app.state.db.insert_result(
        BenchmarkResult(**common, backend_id="first", metadata={"stdout_tail": "first output"})
    )
    app.state.db.insert_result(
        BenchmarkResult(**common, backend_id="second", metadata={"stdout_tail": "second output"})
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/plot-logs/download",
            params={
                "job_id": "job-shared",
                "backend_id": "first",
                "suite": "benchmark",
                "operator_id": "operator",
            },
        )

    assert response.status_code == 200
    assert "first output" in response.text
    assert "second output" not in response.text
    assert "preprocess_ms=3" in response.text
    assert "solve_runtime_ms=1" in response.text


def test_succeeded_spmv_result_download_uses_canonical_csv(tmp_path):
    export_root = tmp_path / "exports"
    app = create_app(
        str(tmp_path / "perf.sqlite"),
        str(tmp_path / "contest.sqlite"),
        result_exports_path=str(export_root),
    )
    job = JobRecord(
        job_id="publish-job",
        generator_id="web-client",
        backends=["rtx5090-workstation"],
        suites=["spmv"],
        dataset_id="suitesparse_sample_100",
        operator_ids=["spmv.csr.fp32"],
        kernels=[KernelArtifact(
            name="SELL C32",
            source="candidate",
            metadata={"operator_id": "spmv.csr.fp32", "base_format": "sell"},
        )],
        status=JobStatus.succeeded,
    )
    app.state.db.upsert_job(job)
    app.state.db.insert_result(BenchmarkResult(
        job_id=job.job_id,
        generator_id=job.generator_id,
        backend_id="rtx5090-workstation",
        backend_kind="gpu",
        suite="spmv",
        operator_id="spmv.csr.fp32",
        operator_name="CSR SpMV FP32",
        matrix_id="group/matrix",
        matrix_name="matrix",
        rows=10,
        cols=10,
        nnz=20,
        kernel_name="SELL C32",
        preprocess_ms=2.0,
        runtime_ms=0.5,
        gflops=0.00008,
        arithmetic_intensity=0.0,
        metadata={"operations": 40, "dtype": "fp32"},
    ))

    exported = app.state.result_exporter.export_job(job)
    assert len(exported) == 1
    canonical_path = Path(exported[0])
    canonical_path.write_text(
        canonical_path.read_text(encoding="utf-8").replace("SELL C32", "stored-file"),
        encoding="utf-8",
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/results.csv",
            params={
                "job_id": job.job_id,
                "backend_id": "rtx5090-workstation",
                "suite": "spmv",
                "operator_id": "spmv.csr.fp32",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "base_format" in response.text.splitlines()[0]
    assert "stored-file" in response.text
    assert "group/matrix" in response.text

from __future__ import annotations

import json
import csv
import io
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse

from .backends import backend_registry_from_config
from .artifacts import SourceArchive
from .benchmark import benchmark_registry_from_config
from .config import load_service_config
from .contest import ContestDatabase
from .datasets import dataset_registry_from_config
from .database import PerfDatabase
from .exports import LocalResultExporter
from .leaderboard import LeaderboardPublisher, make_submission
from .models import (
    ContestCreateRequest,
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
    LeaderboardPublishRequest,
)
from .scheduler import Scheduler
from web.routes import install_web_ui


def create_app(
    db_path: str | None = None,
    contest_db_path: str | None = None,
    config_path: str | None = None,
    result_exports_path: str | None = None,
    source_submissions_path: str | None = None,
) -> FastAPI:
    service = load_service_config(config_path)
    db = PerfDatabase(db_path or service.resolve(service.database))
    contest_db = ContestDatabase(
        contest_db_path or service.resolve(service.contest_database)
    )
    backends = backend_registry_from_config(service.resolve(service.workers))
    datasets = dataset_registry_from_config(service.resolve(service.datasets))
    benchmarks = benchmark_registry_from_config(service.resolve(service.benchmarks))
    result_exporter = LocalResultExporter(
        service.resolve(result_exports_path or service.result_exports),
        db,
        backends,
        benchmarks,
    )
    source_root = (
        Path(source_submissions_path)
        if source_submissions_path
        else Path(db_path).parent / "source"
        if db_path
        else service.resolve(service.source_submissions)
    )
    source_archive = SourceArchive(
        source_root,
        lambda kernel: benchmarks.for_operator(
            str(kernel.metadata["operator_id"])
        ).source_package(kernel),
    )
    scheduler = Scheduler(
        db,
        backends,
        benchmarks,
        datasets,
        contest_db,
        on_job_finished=result_exporter.export_job,
        on_job_submitted=source_archive.archive_job,
    )
    leaderboard = LeaderboardPublisher(service.leaderboard)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        scheduler.start()
        try:
            yield
        finally:
            scheduler.stop()

    app = FastAPI(
        title="KernelPerf",
        version="0.1.0",
        description="Unified test service for AI-generated kernels.",
        lifespan=lifespan,
    )
    app.state.db = db
    app.state.contest_db = contest_db
    app.state.backends = backends
    app.state.datasets = datasets
    app.state.benchmarks = benchmarks
    app.state.suites = benchmarks
    app.state.scheduler = scheduler
    app.state.result_exporter = result_exporter
    app.state.source_archive = source_archive
    app.state.leaderboard = leaderboard

    install_web_ui(app, service.resolve(service.web), contest_db)

    def enrich_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        for row in rows:
            metadata = row.get("metadata") or {}
            dtype = str(metadata.get("dtype", ""))
            if not dtype:
                try:
                    dtype = next(
                        operator.dtype
                        for operator in benchmarks.get(str(row["suite"])).operators()
                        if operator.op_id == row["operator_id"]
                    )
                except (KeyError, StopIteration):
                    dtype = "unknown"
            try:
                backend = backends.get(str(row["backend_id"])).info()
                peak = backend.peak_gflops_by_dtype.get(dtype, backend.peak_gflops)
            except KeyError:
                peak = float(metadata.get("peak_gflops", 0.0))
            gflops = float(row.get("gflops", 0.0))
            row["dtype"] = dtype
            row["peak_gflops"] = peak
            efficiency = (
                gflops / peak * 100
                if row.get("status") == "pass" and gflops > 0 and peak > 0
                else 0.0
            )
            row["solve_only_efficiency_percent"] = efficiency
            row["base_format"] = str(
                metadata.get("implementation", {}).get("base_format", "unknown")
            )
            implementation = metadata.get("implementation", {})
            row["configuration_id"] = str(implementation.get("configuration_id", ""))
            row["candidate_group"] = str(implementation.get("candidate_group", ""))
            row["selection_role"] = str(implementation.get("selection_role", "candidate"))
            job = scheduler.get_job(str(row["job_id"]))
            row["dataset_id"] = job.dataset_id if job and job.dataset_id else "none"
        return rows

    @app.get("/api/v1/backends")
    def list_backends() -> list[dict[str, object]]:
        return [backend.model_dump() for backend in backends.list()]

    @app.get("/api/v1/suites")
    def list_suites() -> list[dict[str, object]]:
        return benchmarks.list()

    @app.get("/api/v1/benchmarks")
    def list_benchmarks() -> list[dict[str, object]]:
        return benchmarks.list()

    @app.get("/api/v1/datasets")
    def list_datasets() -> list[dict[str, object]]:
        return datasets.list()

    @app.get("/api/v1/leaderboard/config")
    def leaderboard_config() -> dict[str, object]:
        return {
            "enabled": leaderboard.enabled,
            "public_url": leaderboard.public_url,
            "branch_url": leaderboard.branch_url,
            "branch": service.leaderboard.get("branch", ""),
        }

    @app.get("/api/v1/contests")
    def list_contests() -> list[dict[str, object]]:
        return contest_db.list_contests()

    @app.post("/api/v1/contests", status_code=201)
    def create_contest(
        request: ContestCreateRequest,
        http_request: Request,
    ) -> dict[str, object]:
        configured_token = os.environ.get("KERNELPERF_ADMIN_TOKEN", "")
        supplied_token = http_request.headers.get("x-kernelperf-admin-token", "")
        if (
            not configured_token
            or not supplied_token
            or not secrets.compare_digest(configured_token, supplied_token)
        ):
            raise HTTPException(
                status_code=403,
                detail="an administrator token is required to create contests",
            )
        try:
            backends.get(request.backend_id)
            datasets.get(request.dataset_id)
            benchmark = benchmarks.get(request.suite)
            operators = benchmark.operators()
            available = {operator.op_id: operator for operator in operators}
            if request.all_operators:
                selected_operator_ids = [
                    operator.op_id
                    for operator in operators
                    if operator.metadata.get("contest_enabled", True)
                ]
                if not selected_operator_ids:
                    raise ValueError(
                        f"Suite {request.suite!r} has no contest-enabled operators"
                    )
            else:
                selected_operator_ids = request.operator_ids or [request.operator_id]
            unknown = [
                operator_id
                for operator_id in selected_operator_ids
                if operator_id not in available
            ]
            if unknown:
                raise ValueError(
                    f"Unknown operators in suite {request.suite!r}: {unknown}"
                )
            resolved = request.model_copy(
                update={
                    "operator_id": None,
                    "operator_ids": selected_operator_ids,
                    "all_operators": False,
                }
            )
            return contest_db.create_contest(resolved)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/v1/contests/{contest_id}")
    def delete_contest(contest_id: str, http_request: Request) -> dict[str, object]:
        configured_token = os.environ.get("KERNELPERF_ADMIN_TOKEN", "")
        supplied_token = http_request.headers.get("x-kernelperf-admin-token", "")
        if (
            not configured_token
            or not supplied_token
            or not secrets.compare_digest(configured_token, supplied_token)
        ):
            raise HTTPException(
                status_code=403,
                detail="an administrator token is required to delete contests",
            )
        if not contest_db.delete_contest(contest_id):
            raise HTTPException(status_code=404, detail="contest not found")
        return {"deleted": True, "contest_id": contest_id}

    @app.get("/api/v1/contests/{contest_id}/leaderboard")
    def contest_leaderboard(contest_id: str) -> dict[str, object]:
        leaderboard = contest_db.leaderboard(contest_id)
        if leaderboard is None:
            raise HTTPException(status_code=404, detail="contest not found")
        return leaderboard

    @app.post("/api/v1/jobs", response_model=JobSubmitResponse, status_code=202)
    def submit_job(request: JobSubmitRequest, http_request: Request) -> JobSubmitResponse:
        effective_suites = request.suites or [benchmarks.default().benchmark_id]
        if request.contest_id is not None:
            contest = contest_db.get_contest(request.contest_id)
            if contest is not None:
                effective_suites = [contest["suite"]]
        try:
            public_submission = all(
                benchmarks.get(benchmark_id).public for benchmark_id in effective_suites
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not public_submission:
            configured_token = os.environ.get("KERNELPERF_ADMIN_TOKEN", "")
            supplied_token = http_request.headers.get("x-kernelperf-admin-token", "")
            if (
                not configured_token
                or not supplied_token
                or not secrets.compare_digest(configured_token, supplied_token)
            ):
                raise HTTPException(
                    status_code=403,
                    detail="an administrator token is required for private benchmarks",
                )
        try:
            job, queue_position = scheduler.submit(request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JobSubmitResponse(
            accepted=True,
            job_id=job.job_id,
            status_url=f"/api/v1/jobs/{job.job_id}",
            queue_position=queue_position,
            message="job accepted",
        )

    @app.get("/api/v1/jobs")
    def list_jobs(status: JobStatus | None = None) -> list[dict[str, object]]:
        return scheduler.db.list_jobs(status=status, limit=200)

    @app.get("/api/v1/job-logs/download", response_class=PlainTextResponse)
    def download_job_logs(job_id: list[str] = Query(...)) -> PlainTextResponse:
        sections = []
        for selected_job_id in dict.fromkeys(job_id):
            if scheduler.get_job(selected_job_id) is None:
                raise HTTPException(status_code=404, detail=f"job not found: {selected_job_id}")
            lines = [f"=== job {selected_job_id} ==="]
            for entry in db.list_job_logs(selected_job_id):
                source = entry["worker_id"] or "scheduler"
                lines.append(f'{entry["timestamp"]} [{source}] {entry["message"]}')
            sections.append("\n".join(lines))
        content = "\n\n".join(sections) + "\n"
        return PlainTextResponse(
            content,
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'attachment; filename="kernelperf-task.log"',
            },
        )

    @app.get("/api/v1/plot-logs/download", response_class=PlainTextResponse)
    def download_plot_logs(
        job_id: list[str] = Query(...),
        backend_id: str = Query(...),
        suite: str = Query(...),
        operator_id: str = Query(...),
    ) -> PlainTextResponse:
        rows = db.query_results(
            job_ids=list(dict.fromkeys(job_id)),
            backend_ids=[backend_id],
            suites=[suite],
            operator_ids=[operator_id],
        )
        if not rows:
            raise HTTPException(status_code=404, detail="no results found for plot")

        lines = [f"=== plot backend={backend_id} suite={suite} operator={operator_id} ==="]
        for row in rows:
            lines.extend([
                "",
                f'--- job={row["job_id"]} kernel={row["kernel_name"]} matrix={row["matrix_name"]} ---',
                f'{row["timestamp"]} status={row["status"]} '
                f'preprocess_ms={row["preprocess_ms"]:.9g} '
                f'solve_runtime_ms={row["runtime_ms"]:.9g} gflops={row["gflops"]:.9g}',
            ])
            metadata = row["metadata"]
            stdout = str(metadata.get("stdout_tail", "")).rstrip()
            stderr = str(metadata.get("stderr_tail", "")).rstrip()
            if stdout:
                lines.extend(["stdout:", stdout])
            if stderr:
                lines.extend(["stderr:", stderr])
            details = {key: value for key, value in metadata.items() if key not in {"stdout_tail", "stderr_tail"}}
            if details:
                lines.append(f"metadata: {json.dumps(details, ensure_ascii=False, sort_keys=True)}")

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{backend_id}_{suite}_{operator_id}")
        return PlainTextResponse(
            "\n".join(lines) + "\n",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="{safe_name}.log"',
            },
        )

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        job = scheduler.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump()

    @app.get("/api/v1/jobs/{job_id}/logs")
    def get_job_logs(job_id: str) -> dict[str, object]:
        if scheduler.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, "logs": db.list_job_logs(job_id)}

    @app.get("/api/v1/queue")
    def get_queue() -> dict[str, object]:
        return scheduler.queue_snapshot()

    @app.get("/api/v1/workers")
    def list_workers() -> list[dict[str, object]]:
        return scheduler.list_workers()

    @app.get("/api/v1/results")
    def query_results(
        job_id: list[str] | None = Query(default=None),
        backend_id: list[str] | None = Query(default=None),
        suite: list[str] | None = Query(default=None),
    ) -> list[dict[str, object]]:
        return enrich_results(
            db.query_results(job_ids=job_id, backend_ids=backend_id, suites=suite)
        )

    def publication_parts(
        request: LeaderboardPublishRequest,
    ) -> tuple[object, list[dict[str, object]], object, object, object, object]:
        job = scheduler.get_job(request.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != JobStatus.succeeded:
            raise HTTPException(status_code=409, detail="only succeeded jobs can be published")
        try:
            backend = backends.get(request.backend_id).info()
            benchmark = benchmarks.get(request.suite)
            operator = next(
                value for value in benchmark.operators()
                if value.op_id == request.operator_id
            )
            candidates = [
                value for value in job.kernels
                if value.metadata.get("operator_id") == request.operator_id
                and (
                    not request.configuration_id
                    or value.metadata.get("configuration_id") == request.configuration_id
                )
            ]
            if len(candidates) != 1:
                raise ValueError("configuration_id is required for a multi-configuration result")
            kernel = candidates[0]
        except (KeyError, StopIteration, ValueError) as exc:
            raise HTTPException(status_code=400, detail="job result selection is invalid") from exc
        rows = db.query_results(
            job_ids=[request.job_id],
            backend_ids=[request.backend_id],
            suites=[request.suite],
            operator_ids=[request.operator_id],
        )
        if request.configuration_id:
            rows = [
                row for row in rows
                if str((row.get("metadata") or {}).get("implementation", {}).get("configuration_id", ""))
                == request.configuration_id
            ]
        if not rows:
            raise HTTPException(status_code=404, detail="no results found for publication")
        return job, rows, backend, benchmark, operator, kernel

    @app.get("/api/v1/results.csv")
    def download_results_csv(
        job_id: str,
        backend_id: str,
        suite: str,
        operator_id: str,
        configuration_id: str | None = None,
        candidate_group: str | None = None,
    ) -> FileResponse:
        job = scheduler.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status not in {
            JobStatus.succeeded,
            JobStatus.failed,
            JobStatus.cancelled,
        }:
            raise HTTPException(status_code=409, detail="results are not complete")
        try:
            if configuration_id == "per-matrix-best":
                result_path = result_exporter.ensure_best(
                    job, backend_id, suite, operator_id, candidate_group
                )
            else:
                result_path = result_exporter.ensure_selection(
                    job, backend_id, suite, operator_id, configuration_id
                )
        except (KeyError, StopIteration, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="job result selection is invalid"
            ) from exc
        if result_path is None:
            raise HTTPException(status_code=404, detail="no results found")
        return FileResponse(
            result_path,
            media_type="text/csv",
            filename=Path(result_path).name,
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/v1/leaderboard/submissions")
    def publish_results(request: LeaderboardPublishRequest) -> dict[str, object]:
        if request.configuration_id == "per-matrix-best":
            job = scheduler.get_job(request.job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            if job.status != JobStatus.succeeded:
                raise HTTPException(status_code=409, detail="only succeeded jobs can be published")
            path = result_exporter.ensure_best(
                job,
                request.backend_id,
                request.suite,
                request.operator_id,
                request.candidate_group,
            )
            if path is None or not path.is_file():
                raise HTTPException(status_code=404, detail="per-matrix best result not found")
            csv_content = path.read_text(encoding="utf-8")
            first = next(csv.DictReader(io.StringIO(csv_content)), None)
            if first is None:
                raise HTTPException(status_code=404, detail="per-matrix best result is empty")
            entry = {
                key: first.get(key, "")
                for key in (
                    "submission_id", "method_id", "method_name", "configuration_id",
                    "candidate_group", "selection_role", "selected_from", "base_format",
                    "operator_id", "dtype", "backend_id", "hardware", "peak_gflops",
                    "dataset_id", "source_kind",
                )
            }
            entry["peak_gflops"] = float(entry["peak_gflops"])
            entry["created_at"] = first.get("timestamp", "")
            try:
                return leaderboard.publish(
                    str(first["submission_id"]), entry, csv_content, None
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        job, rows, backend, benchmark, operator, kernel = publication_parts(request)
        try:
            submission_id, entry, csv_content = make_submission(
                job=job,
                results=rows,
                backend=backend,
                operator=operator,
                kernel=kernel,
            )
            return leaderboard.publish(
                submission_id,
                entry,
                csv_content,
                benchmark.source_package(kernel),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app

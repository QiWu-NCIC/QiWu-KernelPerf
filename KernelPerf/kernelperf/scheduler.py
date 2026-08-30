from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .backends import BackendRegistry
from .benchmark import BenchmarkRegistry, failure_metadata
from .database import PerfDatabase
from .datasets import DatasetRegistry, NO_DATASET_CASE
from .models import (
    BenchmarkResult,
    CaseStatus,
    JobRecord,
    JobStatus,
    JobSubmitRequest,
    MatrixCase,
    WorkerNode,
    WorkerStatus,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configuration_metadata(kernel: object) -> dict[str, str]:
    metadata = getattr(kernel, "metadata", {})
    return {
        "configuration_id": str(metadata.get("configuration_id", "")).strip(),
        "candidate_group": str(metadata.get("candidate_group", "")).strip(),
        "selection_role": str(metadata.get("selection_role", "candidate")).strip() or "candidate",
    }


@dataclass
class _Task:
    job_id: str
    backend_id: str
    status: str = "queued"
    error: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.job_id, self.backend_id


class Scheduler:
    """Fan jobs out into backend tasks consumed independently by idle workers."""

    def __init__(
        self,
        db: PerfDatabase,
        backends: BackendRegistry,
        benchmarks: BenchmarkRegistry,
        datasets: DatasetRegistry,
        on_job_finished: Callable[[JobRecord], list[str]] | None = None,
        on_job_submitted: Callable[[JobRecord], None] | None = None,
    ) -> None:
        self.db = db
        self.backends = backends
        self.benchmarks = benchmarks
        self.datasets = datasets
        self.on_job_finished = on_job_finished
        self.on_job_submitted = on_job_submitted
        self._jobs: dict[str, JobRecord] = {}
        self._tasks: dict[tuple[str, str], _Task] = {}
        self._queue: deque[tuple[str, str]] = deque()
        self._workers: dict[str, WorkerNode] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._assignments: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._register_workers()
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        """Close jobs whose in-memory task state was lost on a service restart."""
        for status in (JobStatus.queued, JobStatus.running):
            for item in self.db.list_jobs(status=status, limit=10000):
                job = self.db.get_job(str(item["job_id"]))
                if job is None:
                    continue
                job.status = JobStatus.cancelled
                job.finished_at = _now()
                job.error = "scheduler restarted before the job completed"
                self.db.upsert_job(job)
                self.db.append_job_log(
                    job.job_id,
                    _now(),
                    "job cancelled during scheduler restart recovery",
                )

    def _register_workers(self) -> None:
        for backend in self.backends.backends():
            info = backend.info()
            labels = {"kind": info.kind, "vendor": info.vendor, **backend.labels}
            worker = WorkerNode(
                worker_id=backend.worker_id,
                backend_id=backend.backend_id,
                endpoint=backend.endpoint,
                labels=labels,
            )
            self._workers[worker.worker_id] = worker
            self.db.upsert_worker(worker)
        self.db.delete_workers_not_in(set(self._workers))

    def start(self) -> None:
        with self._lock:
            self._stop.clear()
            for worker_id in self._workers:
                thread = self._threads.get(worker_id)
                if thread and thread.is_alive():
                    continue
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    name=f"kernelperf-{worker_id}",
                    daemon=True,
                )
                self._threads[worker_id] = thread
                thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in list(self._threads.values()):
            thread.join(timeout=5)

    def submit(self, request: JobSubmitRequest) -> tuple[JobRecord, int]:
        request = request.model_copy(deep=True)
        if not request.suites:
            request.suites = [self.benchmarks.default().benchmark_id]
        if not request.generator_id:
            raise ValueError("generator_id is required")
        if not request.backends:
            raise ValueError("backends is required")
        request.backends = self.backends.resolve(request.backends)
        for benchmark_id in request.suites:
            self.benchmarks.get(benchmark_id).validate_submission(request)
        self._validate_backend_capabilities(request)
        if request.dataset_id:
            self.datasets.get(request.dataset_id)
        payload = request.model_dump()
        job = JobRecord(**payload)
        tasks = [_Task(job.job_id, backend_id) for backend_id in job.backends]
        if self.on_job_submitted is not None:
            self.on_job_submitted(job)
        with self._lock:
            self._jobs[job.job_id] = job
            self._assignments[job.job_id] = set()
            for task in tasks:
                self._tasks[task.key] = task
                self._queue.append(task.key)
            self.db.upsert_job(job)
            queue_position = len(
                dict.fromkeys(key[0] for key in self._queue)
            )
            self.db.append_job_log(
                job.job_id,
                _now(),
                f"accepted job {job.job_id} with {len(tasks)} backend task(s)",
            )
        return job, queue_position

    def _validate_backend_capabilities(self, request: JobSubmitRequest) -> None:
        for backend_id in request.backends or []:
            backend = self.backends.get(backend_id)
            supported = set(backend.info().supported_languages)
            profiles = backend.spec.get("build_profiles", {})
            for kernel in request.kernels:
                if supported and kernel.language and kernel.language not in supported:
                    raise ValueError(
                        f"backend {backend_id!r} does not support language {kernel.language!r}"
                    )
                profile = kernel.metadata.get("build_profile")
                if profile and profile not in profiles:
                    raise ValueError(
                        f"backend {backend_id!r} does not provide build profile {profile!r}"
                    )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            current = self._jobs.get(job_id)
        return current or self.db.get_job(job_id)

    def list_workers(self) -> list[dict[str, object]]:
        with self._lock:
            for worker in self._workers.values():
                self.db.upsert_worker(worker)
            worker_ids = set(self._workers)
        return self.db.list_workers(worker_ids)

    def queue_snapshot(self) -> dict[str, object]:
        with self._lock:
            queued_tasks = [
                {"job_id": job_id, "backend_id": backend_id}
                for job_id, backend_id in self._queue
            ]
            return {
                "queued_job_ids": list(dict.fromkeys(task["job_id"] for task in queued_tasks)),
                "queued_tasks": queued_tasks,
                "running": [
                    {
                        "worker_id": worker.worker_id,
                        "backend_id": worker.backend_id,
                        "job_id": worker.current_job_id,
                    }
                    for worker in self._workers.values()
                    if worker.current_job_id
                ],
            }

    def _worker_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            task = self._claim_task(worker_id)
            if task is None:
                self._stop.wait(0.2)
                continue
            self._run_task(task, self._workers[worker_id])

    def _claim_task(self, worker_id: str) -> _Task | None:
        worker = self._workers[worker_id]
        with self._lock:
            if worker.status != WorkerStatus.online:
                return None
            selected_index = next(
                (
                    index
                    for index, key in enumerate(self._queue)
                    if key[1] == worker.backend_id
                ),
                None,
            )
            if selected_index is None:
                return None
            key = self._queue[selected_index]
            del self._queue[selected_index]
            task = self._tasks[key]
            task.status = "running"
            worker.status = WorkerStatus.busy
            worker.current_job_id = task.job_id
            worker.last_heartbeat = _now()
            self._assignments[task.job_id].add(worker.worker_id)
            job = self._jobs[task.job_id]
            if job.status == JobStatus.queued:
                job.status = JobStatus.running
                job.started_at = _now()
            job.assigned_worker = ",".join(sorted(self._assignments[task.job_id]))
            self.db.upsert_job(job)
            self._log(worker, f"claimed backend task {task.backend_id} for job {task.job_id}")
            return task

    def _run_task(self, task: _Task, worker: WorkerNode) -> None:
        job = self._jobs[task.job_id]
        backend = self.backends.get(task.backend_id)
        try:
            healthy, message = backend.healthcheck()
            self._log(worker, f"backend healthcheck: {message}")
            if not healthy:
                raise RuntimeError(message)
            for benchmark_id in job.suites:
                benchmark = self.benchmarks.get(benchmark_id)
                operators = benchmark.selected_operators(job.operator_ids)
                matrices: list[MatrixCase] = []
                if job.dataset_id:
                    _, matrices = self.datasets.get(job.dataset_id)
                missing_matrix_ids: set[str] = set()
                if job.matrix_ids:
                    selected = set(job.matrix_ids)
                    matrices = [matrix for matrix in matrices if matrix.matrix_id in selected]
                    missing_matrix_ids = selected - {matrix.matrix_id for matrix in matrices}
                    matrices.extend(
                        MatrixCase(
                            matrix_id=matrix_id,
                            name=matrix_id.rsplit("/", 1)[-1],
                            rows=0,
                            cols=0,
                            nnz=0,
                            dataset_id=job.dataset_id,
                        )
                        for matrix_id in sorted(missing_matrix_ids)
                    )
                requires_external_data = any(
                    operator.metadata.get("requires_matrix") is not False
                    for operator in operators
                )
                if operators and not matrices and requires_external_data:
                    missing_matrix_ids.add("missing-dataset-cases")
                    matrices.append(
                        MatrixCase(
                            matrix_id="missing-dataset-cases",
                            name="missing-dataset-cases",
                            rows=0,
                            cols=0,
                            nnz=0,
                            dataset_id=job.dataset_id,
                        )
                    )
                for kernel in job.kernels:
                    selected_operators = benchmark.operators_for_kernel(kernel, operators)
                    for operator in selected_operators:
                        cases = (
                            [NO_DATASET_CASE]
                            if operator.metadata.get("requires_matrix") is False
                            else matrices
                        )
                        for matrix in cases:
                            self._run_and_store_case(
                                worker,
                                backend,
                                benchmark,
                                job,
                                operator,
                                matrix,
                                kernel,
                                preflight_error=(
                                    FileNotFoundError(
                                        f"dataset case is not registered: {matrix.matrix_id}"
                                    )
                                    if matrix.matrix_id in missing_matrix_ids
                                    else None
                                ),
                            )
            task.status = "succeeded"
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            self._log(worker, f"backend task failed: {exc}")
        finally:
            rows = self.db.query_results(
                job_ids=[job.job_id], backend_ids=[task.backend_id]
            )
            counts = {status.value: 0 for status in CaseStatus}
            for row in rows:
                counts[row["status"]] += 1
            self._log(
                worker,
                f"backend task summary pass={counts['pass']} error={counts['error']} fail={counts['fail']}",
            )
            case_failures = counts[CaseStatus.error.value] + counts[CaseStatus.failed.value]
            if case_failures and task.status == "succeeded":
                task.status = "failed"
                task.error = (
                    f"{case_failures} benchmark case(s) returned error/fail status"
                )
            self._finish_task(task, worker)

    def _finish_task(self, task: _Task, worker: WorkerNode) -> None:
        finished_job: JobRecord | None = None
        with self._lock:
            worker.status = WorkerStatus.online
            worker.current_job_id = None
            worker.last_heartbeat = _now()
            self.db.upsert_worker(worker)
            job = self._jobs[task.job_id]
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
            }:
                return
            tasks = [value for value in self._tasks.values() if value.job_id == task.job_id]
            if any(value.status in {"queued", "running"} for value in tasks):
                self.db.upsert_job(job)
                return
            failures = [value for value in tasks if value.status == "failed"]
            job.status = JobStatus.failed if failures else JobStatus.succeeded
            job.finished_at = _now()
            job.error = "; ".join(
                f"{value.backend_id}: {value.error}" for value in failures
            ) or None
            self.db.upsert_job(job)
            self.db.append_job_log(
                job.job_id,
                _now(),
                f"finished job {job.job_id} with status {job.status.value}",
            )
            finished_job = job.model_copy(deep=True)
        if (
            finished_job is not None
            and finished_job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
            }
            and self.on_job_finished is not None
        ):
            try:
                paths = self.on_job_finished(finished_job)
                for path in paths:
                    self.db.append_job_log(
                        finished_job.job_id,
                        _now(),
                        f"exported result CSV: {path}",
                    )
            except Exception as exc:
                self.db.append_job_log(
                    finished_job.job_id,
                    _now(),
                    f"automatic result CSV export failed: {exc}",
                )

    def _log(self, worker: WorkerNode, message: str) -> None:
        with self._lock:
            timestamp = _now()
            worker.log_tail.append(f"{timestamp} {message}")
            worker.log_tail = worker.log_tail[-100:]
            worker.last_heartbeat = timestamp
            self.db.upsert_worker(worker)
            if worker.current_job_id:
                self.db.append_job_log(
                    worker.current_job_id,
                    timestamp,
                    message,
                    worker.worker_id,
                )

    def _run_and_store_case(
        self,
        worker: WorkerNode,
        backend: object,
        benchmark: object,
        job: JobRecord,
        operator: object,
        matrix: MatrixCase,
        kernel: object,
        preflight_error: Exception | None = None,
    ) -> None:
        info = backend.info()
        try:
            if preflight_error is not None:
                raise preflight_error
            result_batch = benchmark.run_case(backend, job, operator, matrix, kernel)
        except Exception as exc:
            result_batch = BenchmarkResult(
                job_id=job.job_id,
                generator_id=job.generator_id,
                backend_id=info.backend_id,
                backend_kind=info.kind,
                suite=benchmark.benchmark_id,
                operator_id=operator.op_id,
                operator_name=operator.name,
                matrix_id=matrix.matrix_id,
                matrix_name=matrix.name,
                rows=matrix.rows,
                cols=matrix.cols,
                nnz=matrix.nnz,
                kernel_name=kernel.name,
                status=CaseStatus.failed,
                runtime_ms=0.0,
                gflops=0.0,
                arithmetic_intensity=0.0,
                metadata={
                    "mode": "failed",
                    "implementation": {
                        "kind": kernel.kind,
                        "base_format": kernel.metadata.get("base_format", "unknown"),
                        **_configuration_metadata(kernel),
                    },
                    **failure_metadata(exc),
                    "validation": {
                        "method": "standard-program-comparison",
                        "failure": "case did not complete",
                    },
                },
            )
        results = result_batch if isinstance(result_batch, list) else [result_batch]
        for result in results:
            self.db.insert_result(result)
            detail = (
                f"preprocess={result.preprocess_ms:.6g} ms, "
                f"solve={result.runtime_ms:.6g} ms, {result.gflops:.6g} GFLOPS"
                if result.status == CaseStatus.passed
                else (
                    f"preprocess={result.preprocess_ms:.6g} ms, "
                    f"solve={result.runtime_ms:.6g} ms, validation error"
                    if result.status == CaseStatus.error
                    else str(
                        result.metadata.get("error")
                        or result.metadata.get("validation", {}).get("evaluation_status")
                        or result.status.value
                    )
                )
            )
            self._log(
                worker,
                f"result {info.backend_id}/{benchmark.benchmark_id}/{operator.op_id}/{result.matrix_name}: "
                f"{result.status.value} {detail}",
            )

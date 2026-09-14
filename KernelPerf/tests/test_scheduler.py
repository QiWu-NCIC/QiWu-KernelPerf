from __future__ import annotations

import threading
import time

from kernelperf.backends import Backend, BackendRegistry, SshBackend
from kernelperf.benchmark import Benchmark, BenchmarkRegistry
from kernelperf.database import PerfDatabase
from kernelperf.datasets import DatasetRegistry
from kernelperf.models import (
    BackendInfo,
    BenchmarkResult,
    CaseStatus,
    JobRecord,
    JobStatus,
    DatasetSpec,
    JobSubmitRequest,
    KernelArtifact,
    MatrixCase,
    OperatorSpec,
)
from kernelperf.scheduler import Scheduler


class FakeBackend(Backend):
    def __init__(self, name: str) -> None:
        super().__init__(
            {
                "worker_id": f"worker-{name}",
                "backend_id": name,
                "transport": "local",
                "kind": "test",
                "name": name,
                "vendor": "test",
                "device": "test",
                "endpoint": f"local://{name}",
            }
        )

    def healthcheck(self):
        return True, "healthy"

    def run(self, command, *, cwd, env=None, timeout=None, stdin=None):
        raise AssertionError("the fake benchmark does not execute commands")

    def write_text(self, path, content):
        raise AssertionError("the fake benchmark does not write files")

    def path_exists(self, path, *, executable=False):
        return True


def test_slurm_ssh_backend_wraps_remote_commands_with_module_and_resources():
    backend = SshBackend(
        {
            "worker_id": "worker-h100",
            "backend_id": "h100",
            "transport": "ssh",
            "kind": "nvidia_gpu",
            "name": "H100",
            "endpoint": "ssh://user@login",
            "ssh": {"host": "login", "user": "user", "port": 22},
            "scheduler": {
                "type": "slurm",
                "account": "ncic",
                "partition": "gpu",
                "gres": "gpu:1",
                "cpus_per_task": 32,
                "mem": "64G",
                "time": "04:00:00",
            },
            "environment": {"modules": ["cuda/12.8.2"]},
        }
    )
    command = backend._remote_command(["nvidia-smi", "--query-gpu=name"], "/shared/build", None)
    assert "module load 'cuda/12.8.2'" in command
    assert "cd '/shared/build'" in command
    assert "srun" in command
    assert "'--account' 'ncic'" in command
    assert "'--gres' 'gpu:1'" in command
    assert "--wait=0" in command
    assert "nvidia-smi" in command


def test_slurm_ssh_backend_can_reuse_an_existing_allocation():
    backend = SshBackend(
        {
            "worker_id": "worker-bw1000",
            "backend_id": "bw1000",
            "transport": "ssh",
            "kind": "dcu_gpu",
            "name": "BW1000",
            "endpoint": "ssh://user@login",
            "ssh": {"host": "login", "user": "user"},
            "scheduler": {
                "type": "slurm",
                "allocation_id": "12345",
                "account": "ignored-inside-allocation",
                "gres": "dcu:bw1000:1",
            },
        }
    )

    command = backend._remote_command(["rocminfo"], "/shared/build", None)

    assert "'--jobid' '12345'" in command
    assert "'--overlap'" in command
    assert "--account" not in command
    assert "--gres" not in command

class ConcurrentBenchmark(Benchmark):
    def __init__(self, entered: set[str], lock: threading.Lock, both: threading.Event) -> None:
        super().__init__(
            {
                "benchmark_id": "concurrent",
                "display_name": "Concurrent",
                "operators": [
                    {
                        "op_id": "unit.operator",
                        "name": "Unit operator",
                        "metadata": {"requires_matrix": False},
                    }
                ],
            },
            __import__("pathlib").Path("config"),
        )
        self.entered = entered
        self.lock = lock
        self.both = both

    def run_case(self, backend, job, operator, matrix, kernel):
        with self.lock:
            self.entered.add(backend.backend_id)
            if len(self.entered) == 2:
                self.both.set()
        self.both.wait(timeout=2)
        info = backend.info()
        return BenchmarkResult(
            job_id=job.job_id,
            generator_id=job.generator_id,
            backend_id=info.backend_id,
            backend_kind=info.kind,
            suite=self.benchmark_id,
            operator_id=operator.op_id,
            operator_name=operator.name,
            matrix_id=matrix.matrix_id,
            matrix_name=matrix.name,
            rows=matrix.rows,
            cols=matrix.cols,
            nnz=matrix.nnz,
            kernel_name=kernel.name,
            runtime_ms=1.0,
            gflops=1.0,
            arithmetic_intensity=0.0,
        )


class CaseFailureBenchmark(ConcurrentBenchmark):
    def run_case(self, backend, job, operator, matrix, kernel):
        result = super().run_case(backend, job, operator, matrix, kernel)
        result.status = CaseStatus.failed
        return result


def test_backend_tasks_run_concurrently_and_job_aggregates(tmp_path):
    backends = BackendRegistry()
    backends.register(FakeBackend("first"))
    backends.register(FakeBackend("second"))
    entered: set[str] = set()
    lock = threading.Lock()
    both = threading.Event()
    benchmarks = BenchmarkRegistry()
    benchmarks.register(ConcurrentBenchmark(entered, lock, both))
    datasets = DatasetRegistry()
    datasets.register(DatasetSpec(dataset_id="none", name="none", kind="none"), [])
    db = PerfDatabase(tmp_path / "perf.sqlite")
    finished_jobs = []
    finished_event = threading.Event()

    def on_job_finished(job):
        finished_jobs.append(job)
        finished_event.set()
        return [str(tmp_path / "export.csv")]

    scheduler = Scheduler(
        db,
        backends,
        benchmarks,
        datasets,
        on_job_finished=on_job_finished,
    )
    scheduler.start()
    try:
        job, _ = scheduler.submit(
            JobSubmitRequest(
                generator_id="test",
                backends=["first", "second"],
                suites=["concurrent"],
                dataset_id="none",
                kernels=[KernelArtifact(name="candidate")],
            )
        )
        deadline = time.monotonic() + 3
        while scheduler.get_job(job.job_id).status.value not in {"succeeded", "failed"}:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        scheduler.stop()

    assert both.is_set()
    assert finished_event.is_set()
    assert [finished.job_id for finished in finished_jobs] == [job.job_id]
    assert scheduler.get_job(job.job_id).status.value == "succeeded"
    assert {row["backend_id"] for row in db.query_results(job_ids=[job.job_id])} == {
        "first",
        "second",
    }
    assert scheduler.queue_snapshot()["queued_tasks"] == []
    assert any(
        "exported result CSV" in entry["message"]
        for entry in db.list_job_logs(job.job_id)
    )


def test_idle_worker_consumes_next_waiting_task(tmp_path):
    backends = BackendRegistry()
    backends.register(FakeBackend("only"))
    benchmarks = BenchmarkRegistry()
    ready = threading.Event()
    ready.set()
    benchmarks.register(ConcurrentBenchmark(set(), threading.Lock(), ready))
    datasets = DatasetRegistry()
    datasets.register(DatasetSpec(dataset_id="none", name="none", kind="none"), [])
    scheduler = Scheduler(
        PerfDatabase(tmp_path / "perf.sqlite"), backends, benchmarks, datasets
    )
    scheduler.start()
    try:
        jobs = [
            scheduler.submit(
                JobSubmitRequest(
                    generator_id=f"test-{index}",
                    backends=["only"],
                    suites=["concurrent"],
                    dataset_id="none",
                    kernels=[KernelArtifact(name="candidate")],
                )
            )[0]
            for index in range(2)
        ]
        deadline = time.monotonic() + 5
        while any(scheduler.get_job(job.job_id).status.value not in {"succeeded", "failed"} for job in jobs):
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        scheduler.stop()
    assert all(scheduler.get_job(job.job_id).status.value == "succeeded" for job in jobs)


def test_case_failures_make_job_failed(tmp_path):
    backends = BackendRegistry()
    backends.register(FakeBackend("only"))
    benchmarks = BenchmarkRegistry()
    benchmarks.register(CaseFailureBenchmark(set(), threading.Lock(), threading.Event()))
    datasets = DatasetRegistry()
    datasets.register(DatasetSpec(dataset_id="none", name="none", kind="none"), [])
    scheduler = Scheduler(
        PerfDatabase(tmp_path / "perf.sqlite"), backends, benchmarks, datasets
    )
    scheduler.start()
    try:
        job, _ = scheduler.submit(
            JobSubmitRequest(
                generator_id="case-failure",
                backends=["only"],
                suites=["concurrent"],
                dataset_id="none",
                kernels=[KernelArtifact(name="candidate")],
            )
        )
        deadline = time.monotonic() + 3
        while scheduler.get_job(job.job_id).status == JobStatus.queued:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        while scheduler.get_job(job.job_id).status == JobStatus.running:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        scheduler.stop()
    assert scheduler.get_job(job.job_id).status == JobStatus.failed


def test_scheduler_restart_closes_incomplete_jobs(tmp_path):
    db = PerfDatabase(tmp_path / "perf.sqlite")
    job = JobRecord(
        job_id="interrupted",
        generator_id="test",
        backends=["only"],
        suites=["concurrent"],
        kernels=[KernelArtifact(name="candidate")],
        status=JobStatus.running,
    )
    db.upsert_job(job)
    backends = BackendRegistry()
    backends.register(FakeBackend("only"))
    benchmarks = BenchmarkRegistry()
    benchmarks.register(ConcurrentBenchmark(set(), threading.Lock(), threading.Event()))
    datasets = DatasetRegistry()
    datasets.register(DatasetSpec(dataset_id="none", name="none", kind="none"), [])
    Scheduler(db, backends, benchmarks, datasets)
    recovered = db.get_job("interrupted")
    assert recovered is not None
    assert recovered.status == JobStatus.cancelled

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends import BackendRegistry, backend_registry_from_config
from .benchmark import BenchmarkRegistry, benchmark_registry_from_config
from .config import ServiceConfig, load_service_config
from .database import PerfDatabase
from .datasets import DatasetRegistry, dataset_registry_from_config
from .exports import LocalResultExporter
from .scheduler import Scheduler


@dataclass
class EvaluationRuntime:
    config: ServiceConfig
    db: PerfDatabase
    backends: BackendRegistry
    benchmarks: BenchmarkRegistry
    datasets: DatasetRegistry
    exporter: LocalResultExporter
    scheduler: Scheduler


def create_runtime(
    config_path: str | Path | None = None,
    *,
    database_path: str | Path | None = None,
    result_exports_path: str | Path | None = None,
) -> EvaluationRuntime:
    config = load_service_config(config_path)
    db = PerfDatabase(database_path or config.resolve(config.database))
    backends = backend_registry_from_config(config.resolve(config.workers))
    benchmarks = benchmark_registry_from_config(config.resolve(config.benchmarks))
    datasets = dataset_registry_from_config(config.resolve(config.datasets))
    exporter = LocalResultExporter(
        result_exports_path or config.resolve(config.result_exports),
        db,
        backends,
        benchmarks,
    )
    scheduler = Scheduler(
        db,
        backends,
        benchmarks,
        datasets,
        on_job_finished=exporter.export_job,
    )
    return EvaluationRuntime(
        config=config,
        db=db,
        backends=backends,
        benchmarks=benchmarks,
        datasets=datasets,
        exporter=exporter,
        scheduler=scheduler,
    )


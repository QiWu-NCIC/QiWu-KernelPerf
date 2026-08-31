from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .backends import BackendRegistry
from .benchmark import BenchmarkRegistry
from .database import PerfDatabase
from .results import make_submission, slug
from .models import JobRecord, JobStatus


class LocalResultExporter:
    """Write candidate and per-matrix-best CSVs after a terminal job."""

    def __init__(
        self,
        root: str | Path,
        db: PerfDatabase,
        backends: BackendRegistry,
        benchmarks: BenchmarkRegistry,
    ) -> None:
        self.root = Path(root)
        self.db = db
        self.backends = backends
        self.benchmarks = benchmarks

    @staticmethod
    def _config_id(kernel: Any) -> str:
        return str(kernel.metadata.get("configuration_id", "")).strip()

    def _resolve_selection(
        self,
        job: JobRecord,
        backend_id: str,
        suite: str,
        operator_id: str,
        configuration_id: str | None = None,
    ) -> tuple[object, object, object, Path]:
        backend = self.backends.get(backend_id).info()
        benchmark = self.benchmarks.get(suite)
        operator = next(value for value in benchmark.operators() if value.op_id == operator_id)
        candidates = [
            value for value in job.kernels
            if value.metadata.get("operator_id") == operator_id
            or benchmark.operators_for_kernel(value, [operator])
        ]
        if configuration_id:
            candidates = [value for value in candidates if self._config_id(value) == configuration_id]
        if len(candidates) != 1:
            raise ValueError(
                f"configuration_id is required when {operator_id!r} has multiple configurations"
            )
        kernel = candidates[0]
        identity = configuration_id or self._config_id(kernel) or kernel.name
        dtype = str(operator.dtype)
        method_id = slug(kernel.name)
        dataset_id = slug(job.dataset_id or "none")
        target = self.root / slug(suite) / slug(backend_id) / dataset_id / (
            f"{method_id}-{slug(backend_id)}-{dataset_id}-{dtype}.csv"
        )
        return backend, operator, kernel, target

    def export_selection(
        self,
        job: JobRecord,
        backend_id: str,
        suite: str,
        operator_id: str,
        configuration_id: str | None = None,
    ) -> Path | None:
        backend, operator, kernel, target = self._resolve_selection(
            job, backend_id, suite, operator_id, configuration_id
        )
        rows = self.db.query_results(
            job_ids=[job.job_id], backend_ids=[backend_id], suites=[suite], operator_ids=[operator_id]
        )
        rows = [
            row for row in rows
            if row["kernel_name"] == kernel.name
            and (
                not configuration_id
                or str((row.get("metadata") or {}).get("implementation", {}).get("configuration_id", ""))
                == configuration_id
            )
        ]
        if not rows:
            return None
        _, _, csv_content = make_submission(
            job=job, results=rows, backend=backend, operator=operator, kernel=kernel
        )
        self._atomic_write(target, csv_content)
        return target.resolve()

    def ensure_selection(
        self,
        job: JobRecord,
        backend_id: str,
        suite: str,
        operator_id: str,
        configuration_id: str | None = None,
    ) -> Path | None:
        _, _, _, target = self._resolve_selection(
            job, backend_id, suite, operator_id, configuration_id
        )
        if target.is_file():
            return target.resolve()
        return self.export_selection(job, backend_id, suite, operator_id, configuration_id)

    def _export_best(
        self,
        job: JobRecord,
        backend_id: str,
        suite: str,
        operator: Any,
        kernels: list[Any],
    ) -> Path | None:
        groups = {str(kernel.metadata.get("candidate_group", "")).strip() for kernel in kernels}
        config_ids = {self._config_id(kernel) for kernel in kernels}
        if len(groups) != 1 or not next(iter(groups)) or len(config_ids) < 2 or "" in config_ids:
            return None
        group = next(iter(groups))
        rows = self.db.query_results(
            job_ids=[job.job_id], backend_ids=[backend_id], suites=[suite], operator_ids=[operator.op_id]
        )
        rows = [
            row for row in rows
            if row["status"] == "pass"
            and str((row.get("metadata") or {}).get("implementation", {}).get("candidate_group", "")) == group
            and str((row.get("metadata") or {}).get("implementation", {}).get("selection_role", "candidate")) == "candidate"
        ]
        selected: dict[str, dict[str, Any]] = {}
        for row in rows:
            implementation = (row.get("metadata") or {}).get("implementation", {})
            config_id = str(implementation.get("configuration_id", ""))
            if not config_id:
                continue
            previous = selected.get(str(row["matrix_id"]))
            previous_impl = (previous or {}).get("metadata", {}).get("implementation", {})
            if previous is None or (float(row["runtime_ms"]), config_id) < (
                float(previous["runtime_ms"]), str(previous_impl.get("configuration_id", ""))
            ):
                selected[str(row["matrix_id"])] = row
        if not selected:
            return None
        selected_from = ",".join(sorted(config_ids))
        representative_id = str(
            (next(iter(selected.values())).get("metadata") or {}).get("implementation", {}).get("configuration_id", "")
        )
        representative = next(kernel for kernel in kernels if self._config_id(kernel) == representative_id)
        backend = self.backends.get(backend_id).info()
        method_id = slug(f"{group}-best")
        _, _, csv_content = make_submission(
            job=job,
            results=list(selected.values()),
            backend=backend,
            operator=operator,
            kernel=representative,
            method_id=method_id,
            method_name=f"{group} BEST",
            configuration_id="per-matrix-best",
            candidate_group=group,
            selection_role="best",
            selected_from=selected_from,
            source_kind="derived",
            base_format="auto",
        )
        dataset_id = slug(job.dataset_id or "none")
        target = self.root / slug(suite) / slug(backend_id) / dataset_id / (
            f"{method_id}-{slug(backend_id)}-{dataset_id}-{operator.dtype}.csv"
        )
        self._atomic_write(target, csv_content)
        return target.resolve()

    def ensure_best(
        self,
        job: JobRecord,
        backend_id: str,
        suite: str,
        operator_id: str,
        candidate_group: str | None = None,
    ) -> Path | None:
        benchmark = self.benchmarks.get(suite)
        operator = next(value for value in benchmark.operators() if value.op_id == operator_id)
        kernels = [
            kernel for kernel in job.kernels
            if kernel.metadata.get("operator_id") == operator_id
            and kernel.metadata.get("selection_role", "candidate") == "candidate"
            and (candidate_group is None or kernel.metadata.get("candidate_group") == candidate_group)
        ]
        return self._export_best(job, backend_id, suite, operator, kernels)

    def export_job(self, job: JobRecord) -> list[str]:
        if job.status not in {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}:
            return []
        exported: list[str] = []
        for backend_id in job.backends:
            for suite in job.suites:
                if suite != "spmv":
                    continue
                benchmark = self.benchmarks.get(suite)
                operators = benchmark.selected_operators(job.operator_ids)
                for kernel in job.kernels:
                    for operator in benchmark.operators_for_kernel(kernel, operators):
                        target = self.export_selection(
                            job, backend_id, suite, operator.op_id, self._config_id(kernel) or None
                        )
                        if target is not None:
                            exported.append(str(target))
                for operator in operators:
                    candidates = [
                        kernel for kernel in job.kernels
                        if kernel.metadata.get("operator_id") == operator.op_id
                        and kernel.metadata.get("selection_role", "candidate") == "candidate"
                    ]
                    target = self._export_best(job, backend_id, suite, operator, candidates)
                    if target is not None:
                        exported.append(str(target))
        return exported

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", prefix=f".{target.name}.",
                suffix=".tmp", dir=target.parent, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_path.chmod(0o644)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

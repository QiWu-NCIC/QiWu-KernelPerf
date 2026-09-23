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

    @staticmethod
    def _latest_attempt_per_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the latest passing attempt, or the latest failure if none passed."""
        selected: dict[str, dict[str, Any]] = {}
        for row in rows:
            matrix_id = str(row["matrix_id"])
            previous = selected.get(matrix_id)
            if previous is None:
                selected[matrix_id] = row
                continue
            current_rank = (row["status"] == "pass", str(row["timestamp"]), str(row["result_id"]))
            previous_rank = (
                previous["status"] == "pass",
                str(previous["timestamp"]),
                str(previous["result_id"]),
            )
            if current_rank > previous_rank:
                selected[matrix_id] = row
        return list(selected.values())

    @staticmethod
    def _latest_spmm_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the latest attempt for each SpMM configuration and scope."""
        selected: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
        for row in rows:
            metadata = row.get("metadata") or {}
            implementation = metadata.get("implementation") or {}
            key = (
                str(implementation.get("candidate_group", "")),
                str(implementation.get("configuration_id", row["kernel_name"])),
                str(row["matrix_id"]),
                int(metadata.get("rhs_columns", 0)),
                str(metadata.get("dense_layout", "")),
            )
            previous = selected.get(key)
            current_rank = (str(row["timestamp"]), str(row["result_id"]))
            previous_rank = (
                (str(previous["timestamp"]), str(previous["result_id"]))
                if previous is not None else ("", "")
            )
            if previous is None or current_rank > previous_rank:
                selected[key] = row
        return list(selected.values())

    def _resolve_selection(
        self,
        job: JobRecord,
        backend_id: str,
        suite: str,
        operator_id: str,
        configuration_id: str | None = None,
        kernel_name: str | None = None,
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
        if kernel_name:
            candidates = [value for value in candidates if value.name == kernel_name]
        if len(candidates) != 1:
            raise ValueError(
                f"configuration_id is required when {operator_id!r} has multiple configurations"
            )
        kernel = candidates[0]
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
        kernel_name: str | None = None,
    ) -> Path | list[Path] | None:
        backend, operator, kernel, target = self._resolve_selection(
            job, backend_id, suite, operator_id, configuration_id, kernel_name
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
        if suite == "spmm":
            return self._export_spmm_selection(
                job, backend, operator, kernel, rows
            )
        rows = self._latest_attempt_per_matrix(rows)
        if not rows:
            return None
        _, _, csv_content = make_submission(
            job=job, results=rows, backend=backend, operator=operator, kernel=kernel
        )
        self._atomic_write(target, csv_content)
        return target.resolve()

    def _spmm_target(
        self,
        job: JobRecord,
        backend_id: str,
        operator: Any,
        method_id: str,
        rhs_columns: int,
        dense_layout: str,
    ) -> Path:
        dataset_id = slug(job.dataset_id or "none")
        layout_id = slug(dense_layout)
        return (
            self.root / "spmm" / slug(backend_id) / dataset_id / operator.dtype /
            f"n{rhs_columns}" / layout_id /
            f"{method_id}-{slug(backend_id)}-{dataset_id}-{operator.dtype}-n{rhs_columns}-{layout_id}.csv"
        )

    def _export_spmm_selection(
        self,
        job: JobRecord,
        backend: Any,
        operator: Any,
        kernel: Any,
        rows: list[dict[str, Any]],
    ) -> list[Path]:
        from benchmarks.spmm.results import make_spmm_submission

        grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for row in rows:
            metadata = row.get("metadata") or {}
            key = (int(metadata.get("rhs_columns", 0)), str(metadata.get("dense_layout", "")))
            if key[0] > 0 and key[1]:
                grouped.setdefault(key, []).append(row)
        exported: list[Path] = []
        method_id = slug(kernel.name)
        for (rhs_columns, dense_layout), scoped_rows in sorted(grouped.items()):
            scoped_rows = self._latest_spmm_attempts(scoped_rows)
            _, _, csv_content = make_spmm_submission(
                job=job,
                results=scoped_rows,
                backend=backend,
                operator=operator,
                kernel=kernel,
            )
            target = self._spmm_target(
                job, backend.backend_id, operator, method_id, rhs_columns, dense_layout
            )
            self._atomic_write(target, csv_content)
            exported.append(target.resolve())
        return exported

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
            base_format="manual-selection",
        )
        dataset_id = slug(job.dataset_id or "none")
        target = self.root / slug(suite) / slug(backend_id) / dataset_id / (
            f"{method_id}-{slug(backend_id)}-{dataset_id}-{operator.dtype}.csv"
        )
        self._atomic_write(target, csv_content)
        return target.resolve()

    def _export_spmm_best(
        self,
        job: JobRecord,
        backend_id: str,
        operator: Any,
        kernels: list[Any],
    ) -> list[Path]:
        from benchmarks.spmm.results import make_spmm_submission

        ranked_kernels = [
            kernel for kernel in kernels
            if bool(kernel.metadata.get("public_ranked", True))
        ]
        groups = {str(kernel.metadata.get("candidate_group", "")).strip() for kernel in ranked_kernels}
        config_ids = {self._config_id(kernel) for kernel in ranked_kernels}
        if len(groups) != 1 or not next(iter(groups)) or len(config_ids) < 2 or "" in config_ids:
            return []
        group = next(iter(groups))
        rows = self.db.query_results(
            job_ids=[job.job_id], backend_ids=[backend_id], suites=["spmm"],
            operator_ids=[operator.op_id]
        )
        rows = self._latest_spmm_attempts(rows)
        rows = [
            row for row in rows
            if row["status"] == "pass"
            and (row.get("metadata") or {}).get("ranking_scope", "main") == "main"
            and str((row.get("metadata") or {}).get("implementation", {}).get("candidate_group", "")) == group
            and str((row.get("metadata") or {}).get("implementation", {}).get("configuration_id", "")) in config_ids
        ]
        timing_fields = {
            "solve-only": lambda row: float(row["runtime_ms"]),
            "pre-plus-solve": lambda row: float((row.get("metadata") or {})["pre_plus_solve_ms"]),
            "pre-amortized": lambda row: float((row.get("metadata") or {})["pre_amortized_ms"]),
        }
        backend = self.backends.get(backend_id).info()
        selected_from = ",".join(sorted(config_ids))
        exported: list[Path] = []
        scopes = sorted({
            (
                int((row.get("metadata") or {})["rhs_columns"]),
                str((row.get("metadata") or {})["dense_layout"]),
            )
            for row in rows
        })
        for rhs_columns, dense_layout in scopes:
            scoped_rows = [
                row for row in rows
                if int((row.get("metadata") or {})["rhs_columns"]) == rhs_columns
                and str((row.get("metadata") or {})["dense_layout"]) == dense_layout
            ]
            for selection_metric, timing in timing_fields.items():
                selected: dict[str, dict[str, Any]] = {}
                for row in scoped_rows:
                    implementation = (row.get("metadata") or {}).get("implementation", {})
                    config_id = str(implementation.get("configuration_id", ""))
                    previous = selected.get(str(row["matrix_id"]))
                    previous_config = str(
                        ((previous or {}).get("metadata") or {}).get("implementation", {}).get("configuration_id", "")
                    )
                    if previous is None or (timing(row), config_id) < (timing(previous), previous_config):
                        selected[str(row["matrix_id"])] = row
                if not selected:
                    continue
                representative_id = str(
                    (next(iter(selected.values())).get("metadata") or {})
                    .get("implementation", {}).get("configuration_id", "")
                )
                representative = next(
                    kernel for kernel in ranked_kernels
                    if self._config_id(kernel) == representative_id
                )
                method_id = slug(f"{group}-best-{selection_metric}")
                _, _, csv_content = make_spmm_submission(
                    job=job,
                    results=list(selected.values()),
                    backend=backend,
                    operator=operator,
                    kernel=representative,
                    method_id=method_id,
                    method_name=f"{group} BEST {selection_metric}",
                    configuration_id="per-matrix-best",
                    candidate_group=group,
                    selection_role="best",
                    selection_metric=selection_metric,
                    selected_from=selected_from,
                    source_kind="derived",
                    base_format="manual-selection",
                )
                target = self._spmm_target(
                    job, backend_id, operator, method_id, rhs_columns, dense_layout
                )
                self._atomic_write(target, csv_content)
                exported.append(target.resolve())
        return exported

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
                if suite not in {"spmv", "spmm"}:
                    continue
                benchmark = self.benchmarks.get(suite)
                operators = benchmark.selected_operators(job.operator_ids)
                for kernel in job.kernels:
                    for operator in benchmark.operators_for_kernel(kernel, operators):
                        target = self.export_selection(
                            job,
                            backend_id,
                            suite,
                            operator.op_id,
                            self._config_id(kernel) or None,
                            kernel.name,
                        )
                        if isinstance(target, list):
                            exported.extend(str(path) for path in target)
                        elif target is not None:
                            exported.append(str(target))
                for operator in operators:
                    candidate_groups: dict[str, list[Any]] = {}
                    for kernel in job.kernels:
                        if (
                            kernel.metadata.get("selection_role", "candidate") != "candidate"
                            or not benchmark.operators_for_kernel(kernel, [operator])
                        ):
                            continue
                        group = str(kernel.metadata.get("candidate_group", "")).strip()
                        if group:
                            candidate_groups.setdefault(group, []).append(kernel)
                    for candidates in candidate_groups.values():
                        if suite == "spmm":
                            exported.extend(
                                str(path) for path in self._export_spmm_best(
                                    job, backend_id, operator, candidates
                                )
                            )
                        else:
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

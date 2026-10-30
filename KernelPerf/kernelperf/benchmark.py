from __future__ import annotations

import importlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import BenchmarkResult, JobRecord, KernelArtifact, MatrixCase, OperatorSpec


class BenchmarkDriverError(RuntimeError):
    def __init__(
        self,
        driver: str,
        stage: str,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.driver = driver
        self.stage = stage
        self.stdout = stdout
        self.stderr = stderr


def failure_metadata(error: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if isinstance(error, BenchmarkDriverError):
        metadata.update(
            {
                "driver": error.driver,
                "failure_stage": error.stage,
                "stdout_tail": error.stdout[-4000:],
                "stderr_tail": error.stderr[-4000:],
            }
        )
    metadata["failure_reason"] = failure_reason(metadata)
    return metadata


_DIAGNOSTIC_RE = re.compile(
    r"\b(?:error|failed|failure|exception|out[ -]of[ -]memory|"
    r"illegal memory|invalid argument|unsupported|timed out|timeout)\b",
    re.IGNORECASE,
)
_POSIX_PATH_RE = re.compile(r"(?<![\w])/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]*")
_WINDOWS_PATH_RE = re.compile(r"(?<![\w])[A-Za-z]:[\\/](?:[^\s'\"]+[\\/])*[^\s'\"]*")
_SECRET_RE = re.compile(
    r"(?i)\b(?:password|passwd|token|secret|identity_file|private_key)\s*[=:]\s*[^\s,;]+"
)


def _sanitize_failure_text(value: Any) -> str:
    text = str(value or "").replace("\x1b", "").strip()
    text = _SECRET_RE.sub(lambda match: match.group(0).split("=")[0].split(":")[0] + "=<redacted>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = re.sub(r"Command .+ timed out after ([0-9.]+) seconds", r"benchmark command timed out after \1 seconds", text)
    return text


def _structured_log_reason(metadata: dict[str, Any]) -> str:
    for log_key in ("stderr_tail", "stdout_tail"):
        for line in reversed(str(metadata.get(log_key) or "").splitlines()):
            try:
                record = json.loads(line.strip())
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(record, dict) and record.get("error"):
                log_error = _sanitize_failure_text(record["error"])
                log_stage = _sanitize_failure_text(record.get("stage"))
                return f"{log_stage}: {log_error}" if log_stage else log_error
    return ""


def _text_log_reason(metadata: dict[str, Any]) -> str:
    for log_key in ("stderr_tail", "stdout_tail"):
        lines = str(metadata.get(log_key) or "").splitlines()
        for line in reversed(lines):
            candidate = _sanitize_failure_text(line)
            if re.match(r"^\d+ (?:errors?|warnings?) (?:detected|generated)", candidate, re.IGNORECASE):
                continue
            if candidate and _DIAGNOSTIC_RE.search(candidate):
                return candidate
    return ""


def failure_reason(metadata: dict[str, Any]) -> str:
    """Return a stable, human-readable reason for a failed result."""
    existing = _sanitize_failure_text(metadata.get("failure_reason"))
    if existing and not existing.startswith("historical reason unavailable"):
        return existing[:512]
    validation = metadata.get("validation") or {}
    validation_status = str(
        validation.get("status")
        or validation.get("evaluation_status")
        or metadata.get("validation_status")
        or metadata.get("accuracy_status")
        or ""
    )
    measurement = metadata.get("measurement") or {}
    accuracy_reason = _sanitize_failure_text(metadata.get("accuracy_reason"))
    failed_elements = validation.get("failed_elements", metadata.get("failed_elements", measurement.get("failed_elements", 0)))
    invalid_elements = validation.get("invalid_elements", metadata.get("invalid_elements", measurement.get("invalid_elements", 0)))
    nan_elements = metadata.get("nan_elements", measurement.get("nan_elements", 0))
    inf_elements = metadata.get("inf_elements", measurement.get("inf_elements", 0))
    mismatch_count = measurement.get("mismatch_count", metadata.get("mismatch_count", 0))

    if accuracy_reason:
        return accuracy_reason[:512]

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    failed_count = count(failed_elements)
    invalid_count = count(invalid_elements)
    nan_count = count(nan_elements)
    inf_count = count(inf_elements)
    mismatch_count_value = count(mismatch_count)
    if (
        validation_status in {"invalid-output", "non-finite-output"}
        or invalid_count > 0
        or nan_count > 0
        or inf_count > 0
    ):
        return (
            f"invalid output: invalid_elements={invalid_count}, "
            f"NaN={nan_count}, Inf={inf_count}"
        )[:512]
    if validation_status in {"precision-error", "accuracy-error"} or failed_count > 0 or mismatch_count_value > 0:
        details = []
        if failed_count or mismatch_count_value:
            details.append(f"failed_elements={failed_count or mismatch_count_value}")
        first_row = measurement.get("first_mismatch_row", metadata.get("first_mismatch_row"))
        if first_row not in (None, "", -1, "-1"):
            details.append(f"first_mismatch_row={first_row}")
            for key in ("first_mismatch_absolute_error", "first_mismatch_bound"):
                value = measurement.get(key, metadata.get(key))
                if value not in (None, ""):
                    details.append(f"{key}={value}")
        return ("precision validation failed" + (f": {', '.join(details)}" if details else ""))[:512]
    structured_reason = _structured_log_reason(metadata)
    if structured_reason:
        return structured_reason[:512]

    text_reason = _text_log_reason(metadata)
    if text_reason:
        stage = _sanitize_failure_text(metadata.get("failure_stage"))
        return (f"{stage}: {text_reason}" if stage else text_reason)[:512]

    error = _sanitize_failure_text(metadata.get("error"))
    stage = _sanitize_failure_text(metadata.get("failure_stage"))
    if error and stage:
        return f"{stage}: {error}"[:512]
    if error:
        return error[:512]
    if stage:
        return stage[:512]
    return "historical reason unavailable (not present in retained result metadata or logs)"


class Benchmark(ABC):
    def __init__(self, spec: dict[str, Any], config_dir: Path) -> None:
        self.spec = spec
        self.config_dir = config_dir
        self.benchmark_id = str(spec["benchmark_id"])
        self.display_name = str(spec.get("display_name", self.benchmark_id))
        self.public = bool(spec.get("public", False))
        self.options = dict(spec.get("options", {}))
        template = spec.get("template")
        self.template_path = None
        if template:
            template_path = Path(template)
            self.template_path = (
                template_path
                if template_path.is_absolute()
                else config_dir.parent / template_path
            )

    def operators(self) -> list[OperatorSpec]:
        operators = []
        for raw in self.spec.get("operators", []):
            item = dict(raw)
            item.setdefault("suite", self.benchmark_id)
            operators.append(OperatorSpec.model_validate(item))
        return operators

    def validate_submission(self, request: Any) -> None:
        del request

    def source_package(self, kernel: KernelArtifact) -> dict[str, object] | None:
        from .artifacts import source_snapshot

        return source_snapshot(kernel)

    def selected_operators(self, operator_ids: list[str] | None) -> list[OperatorSpec]:
        operators = self.operators()
        if operator_ids:
            allowed = set(operator_ids)
            operators = [operator for operator in operators if operator.op_id in allowed]
        return operators

    def operators_for_kernel(
        self,
        kernel: KernelArtifact,
        operators: list[OperatorSpec],
    ) -> list[OperatorSpec]:
        target = kernel.metadata.get("operator_id") or kernel.metadata.get("problem_id")
        if target is None:
            return operators
        return [operator for operator in operators if operator.op_id == target]

    @abstractmethod
    def run_case(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
    ) -> BenchmarkResult | list[BenchmarkResult]:
        raise NotImplementedError

    def discovery(self) -> dict[str, Any]:
        return {
            "suite_id": self.benchmark_id,
            "benchmark_id": self.benchmark_id,
            "display_name": self.display_name,
            "public": self.public,
            "default_dataset_id": self.spec.get("default_dataset_id"),
            "operators": [operator.model_dump() for operator in self.operators()],
            "matrices": [],
        }


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._benchmarks: dict[str, Benchmark] = {}

    def register(self, benchmark: Benchmark) -> None:
        if benchmark.benchmark_id in self._benchmarks:
            raise ValueError(f"duplicate benchmark: {benchmark.benchmark_id}")
        self._benchmarks[benchmark.benchmark_id] = benchmark

    def get(self, benchmark_id: str) -> Benchmark:
        if benchmark_id not in self._benchmarks:
            raise KeyError(f"Unknown benchmark: {benchmark_id}")
        return self._benchmarks[benchmark_id]

    def for_operator(self, operator_id: str) -> Benchmark:
        matches = [
            benchmark
            for benchmark in self._benchmarks.values()
            if any(operator.op_id == operator_id for operator in benchmark.operators())
        ]
        if len(matches) != 1:
            raise KeyError(f"operator does not identify one benchmark: {operator_id}")
        return matches[0]

    def list(self) -> list[dict[str, Any]]:
        return [benchmark.discovery() for benchmark in self._benchmarks.values()]

    def default(self) -> Benchmark:
        defaults = [
            benchmark
            for benchmark in self._benchmarks.values()
            if benchmark.spec.get("default", False)
        ]
        if len(defaults) != 1:
            raise ValueError("benchmark config must declare exactly one default benchmark")
        return defaults[0]


def benchmark_registry_from_config(path: str | Path) -> BenchmarkRegistry:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"benchmark config not found: {config_path}")
    registry = BenchmarkRegistry()
    for spec in json.loads(config_path.read_text()):
        module_name, class_name = str(spec["driver"]).split(":", 1)
        driver_class = getattr(importlib.import_module(module_name), class_name)
        registry.register(driver_class(spec, config_path.parent))
    return registry

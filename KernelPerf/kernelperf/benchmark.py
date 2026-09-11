from __future__ import annotations

import importlib
import json
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
    return metadata


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

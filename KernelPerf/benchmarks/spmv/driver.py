from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from kernelperf.artifacts import source_tree
from kernelperf.benchmark import Benchmark, BenchmarkDriverError
from kernelperf.models import (
    BenchmarkResult,
    CaseStatus,
    JobRecord,
    KernelArtifact,
    MatrixCase,
    OperatorSpec,
)
from .package import GENERATED_FILES, make_source_package


_MAIN_DEFINITION = re.compile(r"\bmain\s*\(")
_RESERVED_TYPE_DEFINITION = re.compile(
    r"\b(?:struct|class|enum\s+class)\s+"
    r"(?:QiwuSpmvCsrInput|QiwuSpmvExecutionContext|QiwuSpmvDataType)\b"
)
_COMMENTS_AND_LITERALS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
_LIFECYCLE_ENTRYPOINTS = (
    "qiwu_spmv_preprocess",
    "qiwu_spmv_solve",
    "qiwu_spmv_destroy",
)
_FAILURE_STAGES = {"setup", "preprocess", "warmup", "solve", "validate", "destroy"}
BASE_FORMAT_CHOICES = frozenset({
    "csr",
    "coo",
    "csc",
    "ell",
    "sell",
    "hyb",
    "bsr",
    "dia",
    "auto",
    "unmarked",
})
CONFIG_METADATA_KEYS = frozenset({
    "configuration_id",
    "candidate_group",
    "selection_role",
})


def configuration_metadata(kernel: KernelArtifact) -> dict[str, str]:
    metadata = kernel.metadata
    configuration_id = str(metadata.get("configuration_id", "")).strip()
    candidate_group = str(metadata.get("candidate_group", "")).strip()
    selection_role = str(metadata.get("selection_role", "candidate")).strip() or "candidate"
    return {
        "configuration_id": configuration_id,
        "candidate_group": candidate_group,
        "selection_role": selection_role,
    }


def _json_record(stdout: str, required_key: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and required_key in record:
            return record
    raise ValueError(f"SpMV benchmark produced no JSON object containing {required_key}")


def parse_measurement(stdout: str) -> dict[str, Any]:
    return _json_record(stdout, "runtime_ms")


def parse_failure(stdout: str) -> dict[str, Any] | None:
    try:
        record = _json_record(stdout, "stage")
    except ValueError:
        return None
    return record if record.get("status") == "error" else None


def object_bytes(kernel: KernelArtifact, options: dict[str, Any]) -> bytes:
    if not kernel.object_base64:
        raise ValueError("spmv object submissions require object_base64")
    try:
        content = base64.b64decode(kernel.object_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("spmv object_base64 is not valid base64") from exc
    limit = int(options.get("object_limit_bytes", 16_000_000))
    if not content or len(content) > limit:
        raise ValueError(f"spmv object must be between 1 and {limit} bytes")
    if len(content) < 20 or not content.startswith(b"\x7fELF"):
        raise ValueError("spmv object must be an ELF relocatable object")
    byte_order = "little" if content[5] == 1 else "big" if content[5] == 2 else ""
    if not byte_order or int.from_bytes(content[16:18], byte_order) != 1:
        raise ValueError("spmv object must be an ELF relocatable object")
    return content


def validate_kernel(kernel: KernelArtifact, options: dict[str, Any]) -> None:
    source_limit = int(options["source_limit_bytes"])
    expected_entrypoint = str(options["entrypoint"])
    if kernel.language != options["language"]:
        raise ValueError(f"spmv implementations must use language {options['language']!r}")
    if kernel.entrypoint != expected_entrypoint:
        raise ValueError(f"spmv implementation entrypoint must be {expected_entrypoint!r}")
    if kernel.path is not None:
        raise ValueError("spmv submissions do not allow server-side paths")
    if kernel.compile_options:
        raise ValueError("spmv compile options are controlled by the benchmark driver")
    required_metadata = {"operator_id", "base_format"}
    optional_metadata = {"build_profile", *CONFIG_METADATA_KEYS}
    object_metadata = {"cuda_arch"} if kernel.kind == "object" else set()
    metadata_keys = set(kernel.metadata)
    expected_metadata = required_metadata | object_metadata
    if not expected_metadata.issubset(metadata_keys) or not metadata_keys.issubset(
        required_metadata | optional_metadata | object_metadata
    ):
        raise ValueError(
            "spmv metadata must contain operator_id and base_format and may contain build_profile"
            + (", plus cuda_arch for objects" if kernel.kind == "object" else "")
        )
    build_profile = kernel.metadata.get("build_profile")
    if build_profile is not None and (not isinstance(build_profile, str) or not build_profile.strip()):
        raise ValueError("spmv metadata.build_profile must be a non-empty string")
    config = configuration_metadata(kernel)
    if config["selection_role"] not in {"candidate", "best"}:
        raise ValueError("spmv metadata.selection_role must be candidate or best")
    if config["selection_role"] == "best":
        raise ValueError("submitted kernels must use selection_role=candidate")
    for key in ("configuration_id", "candidate_group"):
        if kernel.metadata.get(key) is not None and not config[key]:
            raise ValueError(f"spmv metadata.{key} must be a non-empty string")
    if not isinstance(kernel.metadata["operator_id"], str) or not kernel.metadata["operator_id"].strip():
        raise ValueError("spmv metadata.operator_id must be a non-empty string")
    base_format = kernel.metadata["base_format"]
    if not isinstance(base_format, str) or base_format not in BASE_FORMAT_CHOICES:
        choices = ", ".join(sorted(BASE_FORMAT_CHOICES))
        raise ValueError(f"spmv metadata.base_format must be one of: {choices}")

    if kernel.kind == "object":
        if kernel.source is not None or kernel.source_files:
            raise ValueError("spmv object submissions cannot also provide source files")
        if kernel.entry_source is not None or kernel.compile_units:
            raise ValueError("spmv object submissions cannot declare source build settings")
        if not isinstance(kernel.metadata["cuda_arch"], str) or not re.fullmatch(
            r"sm_[0-9]{2,3}", kernel.metadata["cuda_arch"]
        ):
            raise ValueError("spmv object metadata.cuda_arch must look like sm_80")
        object_bytes(kernel, options)
        return

    if kernel.object_base64 is not None:
        raise ValueError("spmv source submissions cannot also provide object_base64")
    files, entry_source, compile_units = source_tree(kernel)
    conflicts = sorted({item.path for item in files} & GENERATED_FILES)
    if conflicts:
        raise ValueError(f"source submission uses reserved plugin paths: {conflicts}")
    maximum_files = int(options.get("max_source_files", 512))
    if len(files) > maximum_files:
        raise ValueError(f"spmv source tree exceeds the {maximum_files}-file limit")
    total_bytes = sum(len(item.content.encode()) for item in files)
    if total_bytes > source_limit:
        raise ValueError(f"spmv source tree exceeds the {source_limit}-byte limit")
    if "qiwu/spmv_plugin.cuh" in {item.path for item in files}:
        raise ValueError("qiwu/spmv_plugin.cuh is reserved by the benchmark")
    if not entry_source.endswith(".cu"):
        raise ValueError("entry_source must be a .cu file")
    invalid_units = [
        value for value in compile_units
        if Path(value).suffix.lower() not in {".cu", ".c", ".cc", ".cpp", ".cxx"}
    ]
    if invalid_units:
        raise ValueError(f"unsupported compile_units: {invalid_units}")
    entry_content = next(item.content for item in files if item.path == entry_source)
    if not entry_content.strip():
        raise ValueError("spmv entry_source must not be empty")
    sanitized = _COMMENTS_AND_LITERALS.sub(" ", entry_content)
    if _MAIN_DEFINITION.search(sanitized):
        raise ValueError("spmv submissions must implement the lifecycle, not main()")
    if _RESERVED_TYPE_DEFINITION.search(sanitized):
        raise ValueError("spmv source redefines a reserved contract type")
    # Multi-file adapters commonly keep the lifecycle definitions in a shared
    # header and expose only a small format-selection entry source. Validate
    # the submitted source tree as a unit; the compiler still decides which
    # files are actually reachable from the selected entry/compile units.
    tree_sanitized = _COMMENTS_AND_LITERALS.sub(
        " ", "\n".join(item.content for item in files)
    )
    missing = [
        symbol
        for symbol in _LIFECYCLE_ENTRYPOINTS
        if not re.search(rf"\b{re.escape(symbol)}\s*\(", sanitized)
        and not re.search(rf"\b{re.escape(symbol)}\s*\(", tree_sanitized)
    ]
    if missing:
        raise ValueError(f"spmv source is missing lifecycle entrypoints: {missing}")


class SpmvBenchmark(Benchmark):
    def __init__(self, spec: dict[str, Any], config_dir: Path) -> None:
        super().__init__(spec, config_dir)
        self.driver_id = str(self.options["driver_id"])
        self.contract_path = self.config_dir.parent / str(self.options["contract"])
        self._executables: dict[tuple[str, str, str], str] = {}

    def source_package(self, kernel: KernelArtifact) -> dict[str, object] | None:
        return make_source_package(kernel, self.contract_path)

    def validate_submission(self, request: Any) -> None:
        if request.suites != [self.benchmark_id]:
            raise ValueError(f"the {self.benchmark_id} benchmark cannot be combined with others")
        if not request.dataset_id:
            raise ValueError(f"{self.benchmark_id} submissions require a configured dataset_id")

        configured_operator_ids = [operator.op_id for operator in self.operators()]
        selected_operator_ids = request.operator_ids or configured_operator_ids
        if not selected_operator_ids or len(set(selected_operator_ids)) != len(selected_operator_ids):
            raise ValueError("spmv operator_ids must be unique and non-empty")
        unknown = sorted(set(selected_operator_ids) - set(configured_operator_ids))
        if unknown:
            raise ValueError(f"unsupported spmv operator_ids: {unknown}")

        submitted_operator_ids: list[str] = []
        for kernel in request.kernels:
            if not kernel.language:
                kernel.language = str(self.options["language"])
            if not kernel.entrypoint:
                kernel.entrypoint = str(self.options["entrypoint"])
            validate_kernel(kernel, self.options)
            operator_id = str(kernel.metadata["operator_id"])
            if operator_id not in selected_operator_ids:
                raise ValueError(
                    f"spmv artifact operator_id {operator_id!r} is not selected by the job"
                )
            submitted_operator_ids.append(operator_id)

        identities: list[tuple[str, str]] = []
        for kernel, operator_id in zip(request.kernels, submitted_operator_ids):
            config = configuration_metadata(kernel)
            identities.append((operator_id, config["configuration_id"]))
        if len(set(identities)) != len(identities):
            raise ValueError("spmv submissions contain duplicate operator/configuration pairs")
        by_operator: dict[str, list[KernelArtifact]] = {}
        for kernel, operator_id in zip(request.kernels, submitted_operator_ids):
            by_operator.setdefault(operator_id, []).append(kernel)
        for operator_id, kernels in by_operator.items():
            if len(kernels) > 1:
                groups = {configuration_metadata(kernel)["candidate_group"] for kernel in kernels}
                if not all(configuration_metadata(kernel)["configuration_id"] for kernel in kernels):
                    raise ValueError(
                        f"multiple configurations for {operator_id!r} require configuration_id"
                    )
                if len(groups) != 1 or not next(iter(groups)):
                    raise ValueError(
                        f"multiple configurations for {operator_id!r} require one candidate_group"
                    )
        if set(submitted_operator_ids) != set(selected_operator_ids):
            missing = sorted(set(selected_operator_ids) - set(submitted_operator_ids))
            unexpected = sorted(set(submitted_operator_ids) - set(selected_operator_ids))
            raise ValueError(
                "spmv submissions must cover the selected operator set; "
                f"missing={missing}, unexpected={unexpected}"
            )

    def _source(self, kernel: KernelArtifact, operator: OperatorSpec) -> str:
        validate_kernel(kernel, self.options)
        if kernel.metadata["operator_id"] != operator.op_id:
            raise ValueError(
                f"artifact {kernel.name!r} is bound to {kernel.metadata['operator_id']!r}, "
                f"not {operator.op_id!r}"
            )
        if self.template_path is None:
            raise RuntimeError("spmv benchmark requires a configured template")
        return self.template_path.read_text()

    def _profile_options(self, backend: Any, kernel: KernelArtifact | None) -> tuple[list[str], list[str]]:
        profile_name = str(kernel.metadata.get("build_profile", "")) if kernel else ""
        if not profile_name:
            return [], []
        profile = getattr(backend, "spec", {}).get("build_profiles", {}).get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(f"unknown backend build profile: {profile_name!r}")
        return (
            [str(value) for value in profile.get("compile_options", [])],
            [str(value) for value in profile.get("link_options", [])],
        )

    def _compile_options(
        self,
        backend: Any,
        operator: OperatorSpec,
        kernel: KernelArtifact | None = None,
    ) -> list[str]:
        options = [str(value) for value in self.options["compile_options"]]
        options.extend(self._profile_options(backend, kernel)[0])
        if operator.dtype == "fp64":
            options.append("-DQIWU_SPMV_FP64=1")
        elif operator.dtype != "fp32":
            raise ValueError(f"unsupported spmv dtype: {operator.dtype!r}")
        architecture = getattr(backend, "labels", {}).get("cuda_arch")
        if architecture:
            options.append(f"-arch={architecture}")
        return options

    def _build_key(
        self,
        backend: Any,
        kernel: KernelArtifact,
        operator: OperatorSpec,
    ) -> str:
        digest = hashlib.sha256()
        components = (
            self.driver_id,
            backend.backend_id,
            operator.op_id,
            operator.dtype,
            str(self.options["compiler"]),
            self._source(kernel, operator),
            json.dumps(
                [item.model_dump() for item in source_tree(kernel)[0]],
                sort_keys=True,
            ) if kernel.kind == "source" else "",
            json.dumps(source_tree(kernel)[2]) if kernel.kind == "source" else "",
            hashlib.sha256(
                object_bytes(kernel, self.options) if kernel.kind == "object" else b""
            ).hexdigest(),
            json.dumps(self._compile_options(backend, operator, kernel), sort_keys=True),
            json.dumps(self._profile_options(backend, kernel)[1], sort_keys=True),
            json.dumps(self.options["link_options"], sort_keys=True),
        )
        for component in components:
            digest.update(component.encode())
            digest.update(b"\0")
        return digest.hexdigest()[:24]

    def _executable(
        self,
        backend: Any,
        kernel: KernelArtifact,
        operator: OperatorSpec,
    ) -> str:
        key = self._build_key(backend, kernel, operator)
        cache_key = (backend.backend_id, operator.op_id, key)
        if cache_key in self._executables:
            return self._executables[cache_key]
        if kernel.kind == "object":
            expected_arch = kernel.metadata["cuda_arch"]
            backend_arch = getattr(backend, "labels", {}).get("cuda_arch")
            if backend_arch != expected_arch:
                raise BenchmarkDriverError(
                    self.driver_id,
                    "compile",
                    f"object targets {expected_arch}, but backend targets {backend_arch or 'unknown'}",
                )
        backend_spec = getattr(backend, "spec", {})
        build_root = backend_spec.get(
            "build_root", self.options["build_root"]
        )
        path_type = PurePosixPath if backend_spec.get("transport") == "ssh" else Path
        build_dir = str(path_type(str(build_root)) / key)
        source_filename = str(self.options.get("source_filename", "benchmark.cu"))
        contract_filename = str(self.options.get("contract_filename", "qiwu/spmv_plugin.cuh"))
        source_path = str(path_type(build_dir) / source_filename)
        object_path = str(path_type(build_dir) / "candidate.o")
        executable = str(path_type(build_dir) / "benchmark")
        if not backend.path_exists(executable, executable=True):
            backend.write_text(source_path, self._source(kernel, operator))
            object_inputs: list[str] = []
            source_inputs: list[str] = []
            source_root = path_type(build_dir) / "source"
            backend.write_text(
                str(source_root / PurePosixPath(contract_filename)),
                self.contract_path.read_text(),
            )
            if kernel.kind == "object":
                backend.write_bytes(object_path, object_bytes(kernel, self.options))
                object_inputs.append(object_path)
            else:
                files, entry_source, compile_units = source_tree(kernel)
                for item in files:
                    backend.write_text(str(source_root / PurePosixPath(item.path)), item.content)
                source_inputs.append(str(source_root / PurePosixPath(entry_source)))
                source_inputs.extend(str(source_root / PurePosixPath(value)) for value in compile_units)
            command = [
                str(self.options["compiler"]),
                *self._compile_options(backend, operator, kernel),
                *(
                    [str(self.options.get("relocatable_option", "-rdc=true"))]
                    if source_inputs and self.options.get("relocatable_option", "-rdc=true")
                    else []
                ),
                "-I",
                str(path_type(build_dir) / "source"),
                source_path,
                *source_inputs,
                *object_inputs,
                *self._profile_options(backend, kernel)[1],
                *[str(value) for value in self.options["link_options"]],
                "-o",
                executable,
            ]
            try:
                completed = backend.run(
                    command,
                    cwd=build_dir,
                    timeout=int(self.options["compile_timeout_seconds"]),
                )
            except Exception as exc:
                raise BenchmarkDriverError(self.driver_id, "compile", str(exc)) from exc
            if completed.returncode != 0:
                raise BenchmarkDriverError(
                    self.driver_id,
                    "compile",
                    f"implementation compilation failed with returncode={completed.returncode}",
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
        self._executables[cache_key] = executable
        return executable

    def _matrix_path(self, backend: Any, matrix: MatrixCase) -> str:
        if not matrix.local_path:
            raise FileNotFoundError(f"dataset path is not configured for {matrix.matrix_id}")
        configured = Path(matrix.local_path).expanduser()
        candidate = configured if configured.suffix.lower() == ".mtx" else configured / f"{matrix.name}.mtx"
        if not backend.path_exists(str(candidate)):
            raise FileNotFoundError(f"dataset file not found for {matrix.matrix_id}: {candidate}")
        return str(candidate)

    def run_case(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
    ) -> BenchmarkResult:
        executable = self._executable(backend, kernel, operator)
        matrix_path = self._matrix_path(backend, matrix)
        driver_config = operator.metadata.get("driver_config", {})
        env = {
            "KERNELPERF_MATRIX_PATH": matrix_path,
            "KERNELPERF_MATRIX_ROWS": str(matrix.rows),
            "KERNELPERF_MATRIX_COLS": str(matrix.cols),
            "KERNELPERF_MATRIX_NNZ": str(matrix.nnz),
            "KERNELPERF_WARMUP": str(driver_config["warmup"]),
            "KERNELPERF_ITERATIONS": str(driver_config["iterations"]),
            "KERNELPERF_VALIDATION_TOLERANCE": str(driver_config["validation_tolerance"]),
        }
        try:
            completed = backend.run(
                [executable],
                cwd=str(
                    (PurePosixPath(executable) if getattr(backend, "spec", {}).get("transport") == "ssh" else Path(executable)).parent
                ),
                env=env,
                timeout=int(self.options["run_timeout_seconds"]),
            )
        except Exception as exc:
            raise BenchmarkDriverError(self.driver_id, "run", str(exc)) from exc
        if completed.returncode != 0:
            failure = parse_failure(completed.stdout)
            stage = str(failure.get("stage", "run")) if failure else "run"
            message = str(failure.get("error")) if failure else (
                f"benchmark failed for {matrix.matrix_id} with returncode={completed.returncode}"
            )
            if stage not in _FAILURE_STAGES:
                stage = "run"
            raise BenchmarkDriverError(
                self.driver_id,
                stage,
                message,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            measurement = parse_measurement(completed.stdout)
            preprocess_ms = float(measurement["preprocess_ms"])
            runtime_ms = float(measurement["runtime_ms"])
            if preprocess_ms < 0:
                raise ValueError(f"invalid preprocess_ms={preprocess_ms}")
            if runtime_ms <= 0:
                raise ValueError(f"invalid runtime_ms={runtime_ms}")
            if not isinstance(measurement.get("valid"), bool):
                raise ValueError("benchmark did not report boolean valid")
            actual_nnz = int(
                measurement.get("metadata", {}).get("actual_nnz", matrix.nnz)
            )
            if actual_nnz <= 0:
                raise ValueError(f"invalid actual_nnz={actual_nnz}")
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkDriverError(
                self.driver_id,
                "result-parse",
                str(exc),
                stdout=completed.stdout,
                stderr=completed.stderr,
            ) from exc
        status = CaseStatus.passed if measurement["valid"] else CaseStatus.error
        operations = float(measurement.get("operations", 2 * matrix.nnz))
        info = backend.info()
        algorithm = driver_config.get("algorithm")
        peak_gflops = info.peak_gflops_by_dtype.get(operator.dtype, info.peak_gflops)
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
            nnz=actual_nnz,
            kernel_name=kernel.name,
            status=status,
            preprocess_ms=preprocess_ms,
            runtime_ms=runtime_ms,
            gflops=(operations / (runtime_ms * 1e6) if status == CaseStatus.passed else 0.0),
            arithmetic_intensity=float(measurement.get("arithmetic_intensity", 0.0)),
            metadata={
                "driver": self.driver_id,
                "mode": "measured",
                "matrix_path": matrix_path,
                "operations": operations,
                "implementation": {
                    "kind": kernel.kind,
                    "base_format": kernel.metadata["base_format"],
                    **configuration_metadata(kernel),
                },
                "dtype": operator.dtype,
                "peak_gflops": peak_gflops,
                "solve_only_efficiency_percent": (
                    operations / (runtime_ms * 1e6) / peak_gflops * 100
                    if status == CaseStatus.passed and peak_gflops > 0
                    else 0.0
                ),
                "algorithm": algorithm,
                "validation": {
                    "method": str(self.options["validation_method"]),
                    "passed": measurement["valid"],
                },
                "measurement": measurement.get("metadata", {}),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
        )

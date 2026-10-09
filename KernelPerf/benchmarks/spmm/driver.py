from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from kernelperf.artifacts import safe_relative_path, source_tree
from kernelperf.benchmark import Benchmark, BenchmarkDriverError, failure_metadata
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
    r"(?:QiwuSpmmProblem|QiwuSpmmExecutionContext|QiwuSpmmDataType|"
    r"QiwuSpmmDenseLayout|QiwuSpmmOperation)\b"
)
_COMMENTS_AND_LITERALS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
_LIFECYCLE_ENTRYPOINTS = (
    "qiwu_spmm_preprocess",
    "qiwu_spmm_solve",
    "qiwu_spmm_library_version",
    "qiwu_spmm_algorithm",
    "qiwu_spmm_destroy",
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
    "algorithm",
    "preprocess_policy",
    "public_ranked",
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
    raise ValueError(f"SpMM benchmark produced no JSON object containing {required_key}")


def _json_records(stdout: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


class _ReplayBackend:
    def __init__(self, backend: Any, completed: subprocess.CompletedProcess[str]) -> None:
        self._backend = backend
        self._completed = completed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def run(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            self._completed.returncode,
            self._completed.stdout,
            self._completed.stderr,
        )


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
        raise ValueError("spmm object submissions require object_base64")
    try:
        content = base64.b64decode(kernel.object_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("spmm object_base64 is not valid base64") from exc
    limit = int(options.get("object_limit_bytes", 16_000_000))
    if not content or len(content) > limit:
        raise ValueError(f"spmm object must be between 1 and {limit} bytes")
    if len(content) < 20 or not content.startswith(b"\x7fELF"):
        raise ValueError("spmm object must be an ELF relocatable object")
    byte_order = "little" if content[5] == 1 else "big" if content[5] == 2 else ""
    if not byte_order or int.from_bytes(content[16:18], byte_order) != 1:
        raise ValueError("spmm object must be an ELF relocatable object")
    return content


def validate_kernel(kernel: KernelArtifact, options: dict[str, Any]) -> None:
    source_limit = int(options["source_limit_bytes"])
    expected_entrypoint = str(options["entrypoint"])
    supported_languages = options.get("languages") or [options.get("language", "cuda")]
    if kernel.language not in supported_languages:
        raise ValueError(
            f"spmm implementations must use one of the languages: {', '.join(map(str, supported_languages))}"
        )
    if kernel.entrypoint != expected_entrypoint:
        raise ValueError(f"spmm implementation entrypoint must be {expected_entrypoint!r}")
    if kernel.path is not None:
        raise ValueError("spmm submissions do not allow server-side paths")
    if kernel.compile_options:
        raise ValueError("spmm compile options are controlled by the benchmark driver")
    required_metadata = {"operator_id", "base_format"}
    optional_metadata = {"build_profile", *CONFIG_METADATA_KEYS}
    object_arch_key = "hip_arch" if kernel.language == "hip" else "cuda_arch"
    object_metadata = {object_arch_key} if kernel.kind == "object" else set()
    metadata_keys = set(kernel.metadata)
    expected_metadata = required_metadata | object_metadata
    if not expected_metadata.issubset(metadata_keys) or not metadata_keys.issubset(
        required_metadata | optional_metadata | object_metadata
    ):
        raise ValueError(
            "spmm metadata must contain operator_id and base_format and may contain lifecycle metadata"
            + (f", plus {object_arch_key} for objects" if kernel.kind == "object" else "")
        )
    build_profile = kernel.metadata.get("build_profile")
    if build_profile is not None and (not isinstance(build_profile, str) or not build_profile.strip()):
        raise ValueError("spmm metadata.build_profile must be a non-empty string")
    config = configuration_metadata(kernel)
    if config["selection_role"] not in {"candidate", "best"}:
        raise ValueError("spmm metadata.selection_role must be candidate or best")
    if config["selection_role"] == "best":
        raise ValueError("submitted kernels must use selection_role=candidate")
    for key in ("configuration_id", "candidate_group"):
        if kernel.metadata.get(key) is not None and not config[key]:
            raise ValueError(f"spmm metadata.{key} must be a non-empty string")
    if not isinstance(kernel.metadata["operator_id"], str) or not kernel.metadata["operator_id"].strip():
        raise ValueError("spmm metadata.operator_id must be a non-empty string")
    base_format = kernel.metadata["base_format"]
    if not isinstance(base_format, str) or base_format not in BASE_FORMAT_CHOICES:
        choices = ", ".join(sorted(BASE_FORMAT_CHOICES))
        raise ValueError(f"spmm metadata.base_format must be one of: {choices}")

    if kernel.kind == "object":
        if kernel.source is not None or kernel.source_files:
            raise ValueError("spmm object submissions cannot also provide source files")
        if kernel.entry_source is not None or kernel.compile_units:
            raise ValueError("spmm object submissions cannot declare source build settings")
        architecture = str(kernel.metadata[object_arch_key])
        pattern = r"gfx[0-9a-z]+" if kernel.language == "hip" else r"sm_[0-9]{2,3}"
        if not re.fullmatch(pattern, architecture):
            raise ValueError(f"spmm object metadata.{object_arch_key} has an invalid architecture")
        object_bytes(kernel, options)
        return

    if kernel.object_base64 is not None:
        raise ValueError("spmm source submissions cannot also provide object_base64")
    files, entry_source, compile_units = source_tree(kernel)
    conflicts = sorted({item.path for item in files} & GENERATED_FILES)
    if conflicts:
        raise ValueError(f"source submission uses reserved plugin paths: {conflicts}")
    maximum_files = int(options.get("max_source_files", 512))
    if len(files) > maximum_files:
        raise ValueError(f"spmm source tree exceeds the {maximum_files}-file limit")
    source_paths = {item.path for item in files}
    for directory in kernel.include_dirs:
        safe_directory = safe_relative_path(directory).as_posix()
        if not any(path == safe_directory or path.startswith(safe_directory + "/") for path in source_paths):
            raise ValueError(f"include directory is missing from source tree: {directory}")
    total_bytes = sum(len(item.content.encode()) for item in files)
    if total_bytes > source_limit:
        raise ValueError(f"spmm source tree exceeds the {source_limit}-byte limit")
    if "qiwu/spmm_plugin.cuh" in {item.path for item in files}:
        raise ValueError("qiwu/spmm_plugin.cuh is reserved by the benchmark")
    source_suffixes = {
        "cuda": {".cu", ".cuh", ".cpp", ".cc", ".cxx"},
        "hip": {".hip", ".cu", ".cuh", ".cpp", ".cc", ".cxx"},
    }
    allowed_suffixes = source_suffixes.get(kernel.language, {".cpp", ".cc", ".cxx"})
    if Path(entry_source).suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"entry_source must use one of: {', '.join(sorted(allowed_suffixes))}"
        )
    invalid_units = [
        value for value in compile_units
        if Path(value).suffix.lower() not in allowed_suffixes | {".c"}
    ]
    if invalid_units:
        raise ValueError(f"unsupported compile_units: {invalid_units}")
    entry_content = next(item.content for item in files if item.path == entry_source)
    if not entry_content.strip():
        raise ValueError("spmm entry_source must not be empty")
    sanitized = _COMMENTS_AND_LITERALS.sub(" ", entry_content)
    if _MAIN_DEFINITION.search(sanitized):
        raise ValueError("spmm submissions must implement the lifecycle, not main()")
    if _RESERVED_TYPE_DEFINITION.search(sanitized):
        raise ValueError("spmm source redefines a reserved contract type")
    # Multi-file adapters commonly keep the lifecycle definitions in a shared
    # header and expose only a small format-selection entry source. Validate
    # the submitted source tree as a unit; the compiler still decides which
    # files are actually reachable from the selected entry/compile units.
    tree_sanitized = "\n".join(
        _COMMENTS_AND_LITERALS.sub(" ", item.content) for item in files
    )
    missing = [
        symbol
        for symbol in _LIFECYCLE_ENTRYPOINTS
        if not re.search(rf"\b{re.escape(symbol)}\s*\(", sanitized)
        and not re.search(rf"\b{re.escape(symbol)}\s*\(", tree_sanitized)
    ]
    if missing:
        raise ValueError(f"spmm source is missing lifecycle entrypoints: {missing}")


class SpmmBenchmark(Benchmark):
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
            raise ValueError("spmm operator_ids must be unique and non-empty")
        unknown = sorted(set(selected_operator_ids) - set(configured_operator_ids))
        if unknown:
            raise ValueError(f"unsupported spmm operator_ids: {unknown}")

        submitted_operator_ids: list[str] = []
        for kernel in request.kernels:
            if not kernel.language:
                kernel.language = str(self.options.get("language", "cuda"))
            if not kernel.entrypoint:
                kernel.entrypoint = str(self.options["entrypoint"])
            validate_kernel(kernel, self.options)
            operator_id = str(kernel.metadata["operator_id"])
            if operator_id not in selected_operator_ids:
                raise ValueError(
                    f"spmm artifact operator_id {operator_id!r} is not selected by the job"
                )
            submitted_operator_ids.append(operator_id)

        identities: list[tuple[str, str]] = []
        for kernel, operator_id in zip(request.kernels, submitted_operator_ids):
            config = configuration_metadata(kernel)
            identities.append((operator_id, config["configuration_id"]))
        if len(set(identities)) != len(identities):
            raise ValueError("spmm submissions contain duplicate operator/configuration pairs")
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
                "spmm submissions must cover the selected operator set; "
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
            raise RuntimeError("spmm benchmark requires a configured template")
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
        language = str(kernel.language) if kernel is not None else str(self.options.get("language", "cuda"))
        language_options = self.options.get("compile_options_by_language", {})
        options = [
            str(value)
            for value in language_options.get(language, self.options["compile_options"])
        ]
        options.extend(self._profile_options(backend, kernel)[0])
        if operator.dtype == "fp64":
            options.append("-DQIWU_SPMM_FP64=1")
        elif operator.dtype != "fp32":
            raise ValueError(f"unsupported spmm dtype: {operator.dtype!r}")
        architecture = getattr(backend, "labels", {}).get(
            "hip_arch" if language == "hip" else "cuda_arch"
        )
        if architecture:
            arch_flag = getattr(backend, "spec", {}).get("arch_flag_by_language", {}).get(
                language, "-arch"
            )
            if arch_flag.endswith("="):
                options.append(f"{arch_flag}{architecture}")
            else:
                options.append(f"{arch_flag}={architecture}")
        if language == "hip":
            options.append("-DQIWU_BACKEND_HIP=1")
        return options

    def _compiler(self, backend: Any, kernel: KernelArtifact | None) -> str:
        language = str(kernel.language) if kernel is not None else str(self.options.get("language", "cuda"))
        by_language = getattr(backend, "spec", {}).get("compiler_by_language", {})
        return str(by_language.get(language, self.options["compiler"]))

    def _link_options(self, backend: Any, kernel: KernelArtifact | None) -> list[str]:
        language = str(kernel.language) if kernel is not None else str(self.options.get("language", "cuda"))
        by_language = self.options.get("link_options_by_language", {})
        options = by_language.get(language, self.options["link_options"])
        return [str(value) for value in options]

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
            self._compiler(backend, kernel),
            self._source(kernel, operator),
            json.dumps(
                [item.model_dump() for item in source_tree(kernel)[0]],
                sort_keys=True,
            ) if kernel.kind == "source" else "",
            json.dumps(source_tree(kernel)[1], sort_keys=True) if kernel.kind == "source" else "",
            json.dumps(source_tree(kernel)[2]) if kernel.kind == "source" else "",
            json.dumps(kernel.include_dirs, sort_keys=True) if kernel.kind == "source" else "",
            hashlib.sha256(
                object_bytes(kernel, self.options) if kernel.kind == "object" else b""
            ).hexdigest(),
            json.dumps(self._compile_options(backend, operator, kernel), sort_keys=True),
            json.dumps(self._profile_options(backend, kernel)[1], sort_keys=True),
            json.dumps(self._link_options(backend, kernel), sort_keys=True),
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
            object_arch_key = "hip_arch" if kernel.language == "hip" else "cuda_arch"
            expected_arch = kernel.metadata[object_arch_key]
            backend_arch = getattr(backend, "labels", {}).get(object_arch_key)
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
        is_remote = backend_spec.get("transport") == "ssh"
        path_type = PurePosixPath if is_remote else Path
        build_dir = str(
            path_type(str(build_root)) / key
            if is_remote
            else Path(str(build_root)).expanduser().resolve() / key
        )
        source_filename = str(
            self.options.get("source_filename_by_language", {}).get(
                kernel.language,
                self.options.get("source_filename", "benchmark.cu"),
            )
        )
        contract_filename = str(self.options.get("contract_filename", "qiwu/spmm_plugin.cuh"))
        source_path = str(path_type(build_dir) / source_filename)
        object_path = str(path_type(build_dir) / "candidate.o")
        executable = str(path_type(build_dir) / "benchmark")
        if not backend.path_exists(executable, executable=True):
            object_inputs: list[str] = []
            source_inputs: list[str] = []
            source_root = path_type(build_dir) / "source"
            upload_files = {
                source_filename: self._source(kernel, operator),
                str(PurePosixPath("source") / PurePosixPath(contract_filename)):
                    self.contract_path.read_text(),
            }
            runtime_path = self.contract_path.with_name("gpu_runtime.h")
            upload_files["source/qiwu/gpu_runtime.h"] = runtime_path.read_text()
            write_texts = getattr(backend, "write_texts", None)
            if write_texts is None:
                def write_texts(root: str, values: dict[str, str]) -> None:
                    for relative, content in values.items():
                        backend.write_text(str(path_type(root) / PurePosixPath(relative)), content)
            if kernel.kind == "object":
                write_texts(build_dir, upload_files)
                backend.write_bytes(object_path, object_bytes(kernel, self.options))
                object_inputs.append(object_path)
            else:
                files, entry_source, compile_units = source_tree(kernel)
                for item in files:
                    upload_files[str(PurePosixPath("source") / PurePosixPath(item.path))] = item.content
                write_texts(build_dir, upload_files)
                source_inputs.append(str(source_root / PurePosixPath(entry_source)))
                source_inputs.extend(str(source_root / PurePosixPath(value)) for value in compile_units)
            command = [
                self._compiler(backend, kernel),
                *self._compile_options(backend, operator, kernel),
                *(
                    [str(self.options.get("relocatable_option_by_language", {}).get(kernel.language, self.options.get("relocatable_option", "-rdc=true")))]
                    if source_inputs and self.options.get("relocatable_option_by_language", {}).get(kernel.language, self.options.get("relocatable_option", "-rdc=true"))
                    else []
                ),
                "-I",
                str(path_type(build_dir) / "source"),
                *sum(([
                    "-I", str(path_type(build_dir) / "source" / PurePosixPath(directory))
                ] for directory in kernel.include_dirs), []),
                source_path,
                *source_inputs,
                *object_inputs,
                *self._profile_options(backend, kernel)[1],
                *self._link_options(backend, kernel),
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
        if getattr(backend, "spec", {}).get("transport") == "ssh":
            configured = PurePosixPath(matrix.local_path)
            candidate = configured if configured.suffix.lower() == ".mtx" else configured / f"{matrix.name}.mtx"
        else:
            configured = Path(matrix.local_path).expanduser()
            candidate = configured if configured.suffix.lower() == ".mtx" else configured / f"{matrix.name}.mtx"
        if not backend.path_exists(str(candidate)):
            raise FileNotFoundError(f"dataset file not found for {matrix.matrix_id}: {candidate}")
        return str(candidate)

    def _failed_result(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
        rhs_columns: int,
        error: Exception,
    ) -> BenchmarkResult:
        info = backend.info()
        error_message = str(error).lower()
        validation_status = "runtime-error"
        if "dimension" in error_message or "matrix metadata" in error_message:
            validation_status = "dimension-error"
        elif "not supported" in error_message or "unsupported" in error_message or "not implemented" in error_message:
            validation_status = "unavailable"
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
            status=CaseStatus.failed,
            runtime_ms=0.0,
            gflops=0.0,
            arithmetic_intensity=0.0,
            metadata={
                "mode": "failed",
                "implementation": {
                    "kind": kernel.kind,
                    "base_format": kernel.metadata["base_format"],
                    **configuration_metadata(kernel),
                    "algorithm": str(kernel.metadata.get("algorithm", "")),
                    "preprocess_policy": str(kernel.metadata.get("preprocess_policy", "")),
                    "public_ranked": bool(kernel.metadata.get("public_ranked", True)),
                },
                "dtype": operator.dtype,
                "rhs_columns": rhs_columns,
                "dense_layout": "row-major",
                "op_a": "N",
                "op_b": "N",
                "alpha": 1.0,
                "beta": 0.0,
                "input_seed": int(operator.metadata["driver_config"]["input_seed"]),
                "ranking_scope": "sanity" if rhs_columns == 1 else "main",
                "operations": 2 * matrix.nnz * rhs_columns,
                **failure_metadata(error),
                "validation": {
                    "method": str(self.options["validation_method"]),
                    "passed": False,
                    "evaluation_status": validation_status,
                },
            },
        )

    def _run_rhs_cases(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
        rhs_values: list[int],
    ) -> list[BenchmarkResult]:
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
            "KERNELPERF_VALIDATION_SAFETY_FACTOR": str(driver_config["validation_safety_factor"]),
            "KERNELPERF_RHS_COLUMNS": ",".join(str(value) for value in rhs_values),
            "KERNELPERF_DENSE_LAYOUT": str(driver_config["dense_layout"]),
            "KERNELPERF_OP_A": str(driver_config["op_a"]),
            "KERNELPERF_OP_B": str(driver_config["op_b"]),
            "KERNELPERF_ALPHA": str(driver_config["alpha"]),
            "KERNELPERF_BETA": str(driver_config["beta"]),
            "KERNELPERF_INPUT_SEED": str(driver_config["input_seed"]),
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
        records_by_rhs: dict[int, dict[str, Any]] = {}
        unscoped_failures: list[dict[str, Any]] = []
        for record in _json_records(completed.stdout):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            raw_rhs = metadata.get("rhs_columns", record.get("rhs_columns", 0))
            try:
                rhs_columns = int(raw_rhs)
            except (TypeError, ValueError):
                rhs_columns = 0
            if rhs_columns in rhs_values:
                if rhs_columns in records_by_rhs:
                    raise BenchmarkDriverError(
                        self.driver_id, "result-parse",
                        f"benchmark produced duplicate records for rhs_columns={rhs_columns}",
                        stdout=completed.stdout, stderr=completed.stderr,
                    )
                records_by_rhs[rhs_columns] = record
            elif record.get("status") == "error":
                unscoped_failures.append(record)
        if completed.returncode != 0 and not records_by_rhs:
            failure = unscoped_failures[-1] if unscoped_failures else parse_failure(completed.stdout)
            stage = str((failure or {}).get("stage", "run"))
            message = str((failure or {}).get("error") or f"benchmark failed for {matrix.matrix_id}")
            raise BenchmarkDriverError(
                self.driver_id, stage if stage in _FAILURE_STAGES else "run", message,
                stdout=completed.stdout, stderr=completed.stderr,
            )
        results: list[BenchmarkResult] = []
        for rhs_columns in rhs_values:
            record = records_by_rhs.get(rhs_columns)
            if record is None:
                error = BenchmarkDriverError(
                    self.driver_id, "result-parse",
                    f"benchmark produced no record for rhs_columns={rhs_columns}",
                    stdout=completed.stdout, stderr=completed.stderr,
                )
                results.append(self._failed_result(backend, job, operator, matrix, kernel, rhs_columns, error))
                continue
            replay_returncode = 0 if "runtime_ms" in record else 1
            replay = _ReplayBackend(
                backend,
                subprocess.CompletedProcess(
                    [executable], replay_returncode, json.dumps(record), completed.stderr
                ),
            )
            try:
                results.append(
                    self._run_rhs_case(
                        replay,
                        job,
                        operator,
                        matrix,
                        kernel,
                        rhs_columns,
                    )
                )
            except Exception as error:
                results.append(self._failed_result(backend, job, operator, matrix, kernel, rhs_columns, error))
        return results

    def _run_rhs_case(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
        rhs_columns: int,
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
            "KERNELPERF_VALIDATION_SAFETY_FACTOR": str(driver_config["validation_safety_factor"]),
            "KERNELPERF_RHS_COLUMNS": str(rhs_columns),
            "KERNELPERF_DENSE_LAYOUT": str(driver_config["dense_layout"]),
            "KERNELPERF_OP_A": str(driver_config["op_a"]),
            "KERNELPERF_OP_B": str(driver_config["op_b"]),
            "KERNELPERF_ALPHA": str(driver_config["alpha"]),
            "KERNELPERF_BETA": str(driver_config["beta"]),
            "KERNELPERF_INPUT_SEED": str(driver_config["input_seed"]),
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
            pre_plus_solve_ms = float(measurement["pre_plus_solve_ms"])
            pre_amortized_ms = float(measurement["pre_amortized_ms"])
            if preprocess_ms < 0:
                raise ValueError(f"invalid preprocess_ms={preprocess_ms}")
            if runtime_ms <= 0:
                raise ValueError(f"invalid runtime_ms={runtime_ms}")
            if pre_plus_solve_ms < runtime_ms or pre_amortized_ms < runtime_ms:
                raise ValueError("preprocess-inclusive timings cannot be below solve-only")
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
        operations = float(measurement.get("operations", 2 * matrix.nnz * rhs_columns))
        info = backend.info()
        measurement_metadata = measurement.get("metadata", {})
        measured_rhs = int(measurement_metadata.get("rhs_columns", rhs_columns))
        if measured_rhs != rhs_columns:
            raise BenchmarkDriverError(
                self.driver_id,
                "result-parse",
                f"benchmark reported rhs_columns={measured_rhs}, expected {rhs_columns}",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        algorithm = str(
            measurement_metadata.get("algorithm")
            or kernel.metadata.get("algorithm")
            or kernel.metadata.get("configuration_id", "")
        )
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
                    "algorithm": str(kernel.metadata.get("algorithm", "")),
                    "preprocess_policy": str(kernel.metadata.get("preprocess_policy", "")),
                    "public_ranked": bool(kernel.metadata.get("public_ranked", True)),
                },
                "dtype": operator.dtype,
                "rhs_columns": rhs_columns,
                "dense_layout": str(driver_config["dense_layout"]),
                "op_a": str(driver_config["op_a"]),
                "op_b": str(driver_config["op_b"]),
                "alpha": float(driver_config["alpha"]),
                "beta": float(driver_config["beta"]),
                "input_seed": int(driver_config["input_seed"]),
                "pre_plus_solve_ms": pre_plus_solve_ms,
                "pre_amortized_ms": pre_amortized_ms,
                "peak_gflops": peak_gflops,
                "solve_only_efficiency_percent": (
                    operations / (runtime_ms * 1e6) / peak_gflops * 100
                    if status == CaseStatus.passed and peak_gflops > 0
                    else 0.0
                ),
                "algorithm": algorithm,
                "library_version": str(measurement_metadata.get("library_version", "unknown")),
                "ranking_scope": "sanity" if rhs_columns == 1 else "main",
                "validation": {
                    "method": str(self.options["validation_method"]),
                    "passed": measurement["valid"],
                    "status": str(measurement_metadata.get("validation_status", "pass")),
                    "failed_elements": int(measurement_metadata.get("failed_elements", 0)),
                    "invalid_elements": int(measurement_metadata.get("invalid_elements", 0)),
                },
                "measurement": measurement_metadata,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
        )

    def run_case(
        self,
        backend: Any,
        job: JobRecord,
        operator: OperatorSpec,
        matrix: MatrixCase,
        kernel: KernelArtifact,
    ) -> list[BenchmarkResult]:
        driver_config = operator.metadata.get("driver_config", {})
        rhs_values = [
            *[int(value) for value in driver_config.get("sanity_rhs_columns", [])],
            *[int(value) for value in driver_config["rhs_columns"]],
        ]
        try:
            return self._run_rhs_cases(
                backend, job, operator, matrix, kernel, rhs_values
            )
        except Exception as error:
            return [
                self._failed_result(
                    backend, job, operator, matrix, kernel, rhs_columns, error
                )
                for rhs_columns in rhs_values
            ]

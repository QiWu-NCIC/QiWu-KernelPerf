from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest

from benchmarks.spmv.driver import (
    SpmvBenchmark,
    parse_failure,
    parse_measurement,
    validate_kernel,
)
from kernelperf.benchmark import BenchmarkDriverError, benchmark_registry_from_config, failure_metadata
from kernelperf.models import (
    BackendInfo,
    CaseStatus,
    JobRecord,
    KernelArtifact,
    MatrixCase,
    SourceFile,
)


KERNEL_SOURCE = Path("tests/fixtures/spmv_lifecycle.cu").read_text()


def benchmark() -> SpmvBenchmark:
    return benchmark_registry_from_config("config/benchmarks.json").get("spmv")


def make_kernel(**changes) -> KernelArtifact:
    values = {
        "name": "candidate",
        "language": "cuda",
        "entrypoint": "qiwu_spmv_plugin",
        "source": KERNEL_SOURCE,
        "metadata": {"operator_id": "spmv.csr.fp32", "base_format": "csr"},
    }
    values.update(changes)
    return KernelArtifact(**values)


def test_managed_runner_uses_the_public_plugin_contract():
    driver = benchmark()
    source = driver._source(make_kernel(), driver.operators()[0])

    assert KERNEL_SOURCE.strip() not in source
    assert source.count("int main()") == 1
    assert "#include <qiwu/spmv_plugin.cuh>" in source
    assert "standard_spmv" in source
    assert "cudaEventElapsedTime" in source


def test_managed_runner_does_not_embed_uploaded_source():
    driver = benchmark()
    candidate = "\ufeff" + KERNEL_SOURCE

    source = driver._source(make_kernel(source=candidate), driver.operators()[0])

    assert "\ufeff" not in source
    assert KERNEL_SOURCE.strip() not in source


def test_multi_file_source_tree_compiles_entry_and_selected_units():
    driver = benchmark()
    operator = driver.operators()[0]
    kernel = make_kernel(
        source=None,
        source_files=[
            SourceFile(path="adapter.cu", content=KERNEL_SOURCE),
            SourceFile(path="upstream/include/core.hpp", content="#pragma once\n"),
            SourceFile(path="upstream/core.cu", content="void upstream_core() {}\n"),
            SourceFile(path="upstream/not-selected.cu", content="int main() { return 1; }\n"),
        ],
        entry_source="adapter.cu",
        compile_units=["upstream/core.cu"],
    )
    backend = ObjectBuildBackend()

    validate_kernel(kernel, driver.options)
    driver._executable(backend, kernel, operator)

    normalized_command = [value.replace("\\", "/") for value in backend.command]
    assert any(value.endswith("adapter.cu") for value in normalized_command)
    assert any(value.endswith("upstream/core.cu") for value in normalized_command)
    assert not any(value.endswith("upstream/not-selected.cu") for value in normalized_command)
    assert "-rdc=true" in backend.command
    written = {path.replace("\\", "/"): content for path, content in backend.text_written}
    assert any(path.endswith("source/qiwu/spmv_plugin.cuh") for path in written)


def test_multi_file_source_tree_allows_lifecycle_in_included_adapter_header():
    driver = benchmark()
    kernel = make_kernel(
        source=None,
        source_files=[
            SourceFile(path="variants/csr.cu", content='#include "adapter.cuh"\n'),
            SourceFile(path="adapter.cuh", content=KERNEL_SOURCE),
        ],
        entry_source="variants/csr.cu",
        compile_units=[],
    )
    validate_kernel(kernel, driver.options)


def test_csc_base_format_is_allowed():
    driver = benchmark()
    kernel = make_kernel(metadata={"operator_id": "spmv.csr.fp32", "base_format": "csc"})
    validate_kernel(kernel, driver.options)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"path": "/server/kernel.cu", "source": None}, "server-side paths"),
        ({"source": "int main() { return 0; }"}, "not main"),
        ({"entrypoint": "other"}, "entrypoint"),
        ({"language": "cpp"}, "language"),
        ({"compile_options": ["-O0"]}, "controlled"),
        ({"metadata": {"driver": "custom"}}, "operator_id"),
        ({"metadata": {}}, "operator_id"),
        ({"metadata": {"operator_id": "spmv.csr.fp32", "base_format": "sell-c32"}}, "one of"),
        ({"source": None}, "require source"),
        ({"source": "extern \"C\" void qiwu_spmv_solve() {}"}, "missing lifecycle"),
    ],
)
def test_lifecycle_contract_rejects_unmanaged_fields(changes, message):
    driver = benchmark()
    with pytest.raises(ValueError, match=message):
        validate_kernel(make_kernel(**changes), driver.options)


class BuildBackend:
    backend_id = "rtx5090-workstation"
    labels = {"cuda_arch": "sm_120"}
    spec = {}


def test_build_key_separates_dtype_and_includes_architecture():
    driver = benchmark()
    fp32, fp64 = driver.operators()
    fp32_kernel = make_kernel()
    fp64_kernel = make_kernel(metadata={"operator_id": fp64.op_id, "base_format": "csr"})

    assert driver._build_key(BuildBackend(), fp32_kernel, fp32) != driver._build_key(
        BuildBackend(), fp64_kernel, fp64
    )
    assert "-arch=sm_120" in driver._compile_options(BuildBackend(), fp32)
    assert "-DQIWU_SPMV_FP64=1" in driver._compile_options(BuildBackend(), fp64)


class ObjectBuildBackend:
    backend_id = "a100-server"
    labels = {"cuda_arch": "sm_80"}
    spec = {"build_root": "/shared/kernelperf-build", "transport": "ssh"}

    def __init__(self):
        self.bytes_written = None
        self.command = None
        self.text_written = []

    def path_exists(self, path, *, executable=False):
        return False

    def write_text(self, path, content):
        self.text_written.append((path, content))

    def write_bytes(self, path, content):
        self.bytes_written = (path, content)

    def run(self, command, *, cwd, env=None, timeout=None, stdin=None):
        self.command = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_relocatable_object_is_uploaded_and_linked():
    driver = benchmark()
    operator = driver.operators()[0]
    header = bytearray(b"\x7fELF" + b"\0" * 60)
    header[5] = 1
    header[16:18] = (1).to_bytes(2, "little")
    content = bytes(header)
    kernel = make_kernel(
        kind="object",
        source=None,
        object_base64=base64.b64encode(content).decode(),
        metadata={
            "operator_id": operator.op_id,
            "base_format": "sell",
            "cuda_arch": "sm_80",
        },
    )
    backend = ObjectBuildBackend()

    executable = driver._executable(backend, kernel, operator)

    assert executable.endswith("benchmark")
    assert executable.startswith("/shared/kernelperf-build/")
    assert backend.bytes_written[1] == content
    assert any(value.endswith("candidate.o") for value in backend.command)


def test_object_backend_architecture_is_checked():
    driver = benchmark()
    operator = driver.operators()[0]
    header = bytearray(b"\x7fELF" + b"\0" * 60)
    header[5] = 1
    header[16:18] = (1).to_bytes(2, "little")
    content = base64.b64encode(header).decode()
    wrong_arch = make_kernel(
        kind="object",
        source=None,
        object_base64=content,
        metadata={
            "operator_id": operator.op_id,
            "base_format": "csr",
            "cuda_arch": "sm_90",
        },
    )
    validate_kernel(wrong_arch, driver.options)
    with pytest.raises(BenchmarkDriverError, match="object targets sm_90"):
        driver._executable(ObjectBuildBackend(), wrong_arch, operator)


def test_submission_requires_exact_operator_coverage():
    driver = benchmark()
    request = type(
        "Request",
        (),
        {
            "suites": ["spmv"],
            "dataset_id": "dataset",
            "operator_ids": None,
            "kernels": [make_kernel()],
        },
    )()
    with pytest.raises(ValueError, match="cover the selected operator set"):
        driver.validate_submission(request)


def test_submission_allows_multiple_configurations_for_one_operator():
    driver = benchmark()
    kernels = [
        make_kernel(
            name="cfg-a",
            metadata={
                "operator_id": "spmv.csr.fp32",
                "base_format": "csr",
                "configuration_id": "a",
                "candidate_group": "group",
            },
        ),
        make_kernel(
            name="cfg-b",
            metadata={
                "operator_id": "spmv.csr.fp32",
                "base_format": "csr",
                "configuration_id": "b",
                "candidate_group": "group",
            },
        ),
    ]
    request = type(
        "Request",
        (),
        {
            "suites": ["spmv"],
            "dataset_id": "dataset",
            "operator_ids": ["spmv.csr.fp32"],
            "kernels": kernels,
        },
    )()
    driver.validate_submission(request)


def test_submission_rejects_duplicate_configuration_id():
    driver = benchmark()
    first = make_kernel(
        metadata={
            "operator_id": "spmv.csr.fp32",
            "base_format": "csr",
            "configuration_id": "same",
            "candidate_group": "group",
        }
    )
    second = first.model_copy(deep=True)
    second.name = "other"
    request = type(
        "Request",
        (),
        {
            "suites": ["spmv"],
            "dataset_id": "dataset",
            "operator_ids": ["spmv.csr.fp32"],
            "kernels": [first, second],
        },
    )()
    with pytest.raises(ValueError, match="duplicate operator/configuration"):
        driver.validate_submission(request)


def test_parse_measurement_and_structured_failure_use_last_json_record():
    measurement = parse_measurement(
        'driver log\n{"preprocess_ms":3.5,"runtime_ms":1.25,"operations":5000,"valid":true}\n'
    )
    failure = parse_failure(
        'log\n{"status":"error","stage":"preprocess","error":"bad storage"}\n'
    )
    assert measurement["preprocess_ms"] == 3.5
    assert measurement["runtime_ms"] == 1.25
    assert failure == {"status": "error", "stage": "preprocess", "error": "bad storage"}


class MeasurementBackend:
    backend_id = "worker-backend"

    def __init__(self, valid: bool) -> None:
        self.valid = valid

    def info(self):
        return BackendInfo(
            backend_id=self.backend_id,
            kind="gpu",
            name="GPU",
            vendor="vendor",
            device="device",
            peak_gflops=1,
            memory_bandwidth_gbs=1,
            supported_languages=["cuda"],
        )

    def path_exists(self, path, *, executable=False):
        return True

    def write_text(self, path, content):
        raise AssertionError("cached executable should not be rewritten")

    def run(self, command, *, cwd, env=None, timeout=None, stdin=None):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"preprocess_ms":2.5,"runtime_ms":0.5,"operations":1000000,"valid":'
                + str(self.valid).lower()
                + ',"metadata":{"actual_nnz":125}}\n'
            ),
            stderr="",
        )


def test_measurement_is_classified_and_scored_by_driver(tmp_path):
    driver = benchmark()
    operator = driver.operators()[0]
    job = JobRecord(
        generator_id="test",
        backends=["worker-backend"],
        suites=["spmv"],
        dataset_id="dataset",
        operator_ids=[operator.op_id],
        kernels=[make_kernel()],
    )
    matrix = MatrixCase(
        matrix_id="group/matrix",
        name="matrix",
        rows=10,
        cols=10,
        nnz=100,
        local_path=str(tmp_path / "matrix"),
    )

    passed = driver.run_case(MeasurementBackend(True), job, operator, matrix, make_kernel())
    incorrect = driver.run_case(MeasurementBackend(False), job, operator, matrix, make_kernel())

    assert passed.status == CaseStatus.passed
    assert passed.preprocess_ms == pytest.approx(2.5)
    assert passed.nnz == 125
    assert passed.gflops == pytest.approx(2.0)
    assert incorrect.status == CaseStatus.error
    assert incorrect.gflops == 0.0
    assert passed.metadata["validation"]["method"] == "cpu-csr-reference"


def test_driver_failure_metadata_is_bounded_and_structured():
    metadata = failure_metadata(
        BenchmarkDriverError(
            "spmv-lifecycle",
            "preprocess",
            "candidate failed",
            stdout="driver stdout",
            stderr="driver stderr",
        )
    )

    assert metadata["driver"] == "spmv-lifecycle"
    assert metadata["failure_stage"] == "preprocess"
    assert metadata["stdout_tail"] == "driver stdout"

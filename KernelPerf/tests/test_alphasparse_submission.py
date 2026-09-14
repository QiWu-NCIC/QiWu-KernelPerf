from __future__ import annotations

from pathlib import Path

from kernelperf.benchmark import benchmark_registry_from_config
from kernelperf.models import KernelArtifact, SourceFile


def test_alphasparse_csr_candidates_use_managed_lifecycle():
    root = Path("submissions/spmv/alphasparse")
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in {
            "submission.json", "CMakeLists.txt", "README-QIWU-PLUGIN.md"
        } or "include" in path.parts or "examples" in path.parts:
            continue
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("gb18030")
        files.append(SourceFile(path=path.relative_to(root).as_posix(), content=content))
    driver = benchmark_registry_from_config("config/benchmarks.json").get("spmv")
    operator = driver.operators()[0]
    for name in ("scalar", "vector", "merge", "line_enhance", "flat1", "flat4", "flat8"):
        kernel = KernelArtifact(
            name=f"alphasparse-{name}",
            language="cuda",
            entrypoint="qiwu_spmv_plugin",
            source_files=files,
            entry_source=f"variants/{name}.cu",
            metadata={
                "operator_id": operator.op_id,
                "base_format": "csr",
                "build_profile": "alphasparse-cuda",
            },
        )
        driver.validate_submission(type("Request", (), {
            "suites": ["spmv"],
            "dataset_id": "dataset",
            "operator_ids": [operator.op_id],
            "kernels": [kernel],
        })())
        source = driver._source(kernel, operator)
        assert "int main()" in source
        assert "qiwu_spmv_preprocess" in source
        assert "#include <qiwu/spmv_plugin.cuh>" in source
        variant = next(item.content for item in files if item.path == f"variants/{name}.cu")
        assert "KERNELPERF_ALPHASPARSE_ALGORITHM" in variant

    adapter = next(item.content for item in files if item.path == "adapter.cu")
    assert "clear_output(storage, stream);" in adapter
    assert "if (context && context->reset_output)" in adapter


def test_alphasparse_standalone_cmake_has_a_repository_default_entry():
    cmake = Path("submissions/spmv/alphasparse/CMakeLists.txt").read_text()
    assert 'set(QIWU_PLUGIN_ENTRY "variants/vector.cu")' in cmake


def test_alphasparse_submission_can_select_native_hip():
    from kernelperf.submissions import load_submission_artifacts

    artifacts = load_submission_artifacts(
        Path("submissions/spmv/alphasparse"),
        operator_id="spmv.csr.fp32",
        language="hip",
    )
    assert artifacts
    assert all(artifact.language == "hip" for artifact in artifacts)


def test_hipsparse_submission_selects_hip_name_and_profile():
    from kernelperf.submissions import load_submission_artifacts

    artifacts = load_submission_artifacts(
        Path("submissions/spmv/cusparse"),
        operator_id="spmv.csr.fp32",
        configuration_ids=["csr-default"],
        language="hip",
    )
    assert len(artifacts) == 1
    assert artifacts[0].name == "hipSPARSE CSR DEFAULT"
    assert artifacts[0].metadata["build_profile"] == "hipsparse-hip"

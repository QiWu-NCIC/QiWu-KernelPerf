from __future__ import annotations

from pathlib import Path

from kernelperf.benchmark import benchmark_registry_from_config
from kernelperf.models import KernelArtifact, SourceFile


def test_alphasparse_csr_candidates_use_managed_lifecycle():
    root = Path("submissions/spmv/alphasparse")
    files = [
        SourceFile(path=path.relative_to(root).as_posix(), content=path.read_text(encoding="utf-8"))
        for path in root.rglob("*")
            if path.is_file() and path.name not in {
                "submission.json", "CMakeLists.txt", "README-QIWU-PLUGIN.md"
            } and "include" not in path.parts and "examples" not in path.parts
    ]
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

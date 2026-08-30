from __future__ import annotations

import json

import pytest

from benchmarks.spmv.package import make_source_package
from kernelperf.models import KernelArtifact, SourceFile


ADAPTER = """#include <qiwu/spmv_plugin.cuh>
struct QiwuSpmvStorage {};
extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput*, const QiwuSpmvExecutionContext*, cudaStream_t) { return nullptr; }
extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage*, const QiwuSpmvExecutionContext*, cudaStream_t) {}
extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage*, cudaStream_t) {}
"""


def kernel(*, extra: SourceFile | None = None) -> KernelArtifact:
    files = [SourceFile(path="adapter.cu", content=ADAPTER)]
    if extra:
        files.append(extra)
    return KernelArtifact(
        name="standalone-test",
        kind="source",
        source_files=files,
        entry_source="adapter.cu",
        metadata={"operator_id": "spmv.csr.fp32", "base_format": "csr"},
    )


def test_source_package_contains_a_standalone_build(tmp_path):
    contract = tmp_path / "spmv_plugin.cuh"
    contract.write_text("#pragma once\n")

    package = make_source_package(kernel(), contract)
    assert package is not None
    files = {item["path"]: item["content"] for item in package["files"]}
    assert files["include/qiwu/spmv_plugin.cuh"] == "#pragma once\n"
    assert "add_library(qiwu_spmv_${dtype} STATIC" in files["CMakeLists.txt"]
    assert "qiwu_spmv_preprocess" in files["examples/standalone.cu"]
    assert "plugin.json" not in files
    assert package["plugin"]["kind"] == "qiwu-spmv-source-plugin"
    assert package["plugin"]["files"] == [item["path"] for item in package["files"]]
    assert package["plugin"]["source_sha256"] == package["source_sha256"]
    assert len(package["source_sha256"]) == 64
    assert package["source_sha256"] == make_source_package(kernel(), contract)["source_sha256"]


def test_source_package_rejects_generated_path_collisions(tmp_path):
    contract = tmp_path / "spmv_plugin.cuh"
    contract.write_text("#pragma once\n")
    conflicting = kernel(extra=SourceFile(path="plugin.json", content="{}\n"))

    with pytest.raises(ValueError, match="reserved plugin paths"):
        make_source_package(conflicting, contract)


def test_source_package_indexes_sweep_configurations(tmp_path):
    contract = tmp_path / "spmv_plugin.cuh"
    contract.write_text("#pragma once\n")
    candidate = kernel()
    candidate.source_files.extend([
        SourceFile(path="variants/vector.cu", content='#include "../adapter.cu"\n'),
        SourceFile(
            path="sweep.json",
            content=json.dumps([{
                "configuration_id": "vector",
                "entry_source": "variants/vector.cu",
                "method_name": "Vector",
            }]),
        ),
    ])

    package = make_source_package(candidate, contract)
    assert package is not None
    assert package["plugin"]["configurations"] == [{
        "configuration_id": "vector",
        "entry_source": "variants/vector.cu",
        "method_name": "Vector",
    }]
    cmake = next(item["content"] for item in package["files"] if item["path"] == "CMakeLists.txt")
    assert 'file(READ "${CMAKE_CURRENT_SOURCE_DIR}/plugin.json"' in cmake

from __future__ import annotations

import json

import pytest

from kernelperf.artifacts import SourceArchive, safe_relative_path, source_tree
from kernelperf.models import JobRecord, KernelArtifact, SourceFile


ADAPTER = """
extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput*, const QiwuSpmvExecutionContext*, cudaStream_t);
extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage*, const QiwuSpmvExecutionContext*, cudaStream_t);
extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage*, cudaStream_t);
"""


def multi_file_kernel() -> KernelArtifact:
    return KernelArtifact(
        name="upstream-adapter",
        language="cuda",
        entrypoint="qiwu_spmv_plugin",
        source_files=[
            SourceFile(path="adapter.cu", content=ADAPTER),
            SourceFile(path="upstream/include/spmv.hpp", content="#pragma once\n"),
            SourceFile(path="upstream/src/spmv.cu", content="void upstream_spmv() {}\n"),
        ],
        entry_source="adapter.cu",
        compile_units=["upstream/src/spmv.cu"],
        metadata={"operator_id": "spmv.csr.fp32", "base_format": "csr"},
    )


@pytest.mark.parametrize("path", ["../secret", "/absolute.cu", "dir\\file.cu", "./file.cu"])
def test_source_paths_are_confined_to_the_submission(path):
    with pytest.raises(ValueError, match="source path|unsafe"):
        safe_relative_path(path)


def test_source_tree_normalizes_legacy_single_source():
    kernel = KernelArtifact(name="legacy", source=ADAPTER)

    files, entry_source, compile_units = source_tree(kernel)

    assert [(item.path, item.content) for item in files] == [("adapter.cu", ADAPTER)]
    assert entry_source == "adapter.cu"
    assert compile_units == []


def test_source_archive_keeps_each_job_and_operator_in_its_own_directory(tmp_path):
    kernel = multi_file_kernel()
    job = JobRecord(
        generator_id="test",
        backends=["backend"],
        suites=["spmv"],
        kernels=[kernel],
    )

    SourceArchive(tmp_path).archive_job(job)

    root = tmp_path / job.job_id / "spmv.csr.fp32"
    plugin = json.loads((root / "plugin.json").read_text())
    assert plugin["entry_source"] == "adapter.cu"
    assert plugin["compile_units"] == ["upstream/src/spmv.cu"]
    assert not (root / "manifest.json").exists()
    assert (root / "files/adapter.cu").read_text() == ADAPTER
    assert (root / "files/upstream/src/spmv.cu").is_file()

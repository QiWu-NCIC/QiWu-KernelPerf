from __future__ import annotations

import json

import pytest

from scripts.run_regression import submission_directories, supported_operators


def test_regression_discovers_only_submission_manifests(tmp_path):
    submission = tmp_path / "spmv" / "method"
    submission.mkdir(parents=True)
    (submission / "submission.json").write_text(
        json.dumps({
            "operator_id": "spmv.csr.fp32",
            "supported_operators": ["spmv.csr.fp32", "spmv.csr.fp64"],
        }),
        encoding="utf-8",
    )
    (tmp_path / "spmv" / "templates").mkdir()

    assert submission_directories(tmp_path, []) == [submission]
    assert supported_operators(submission) == ["spmv.csr.fp32", "spmv.csr.fp64"]


def test_regression_rejects_explicit_directory_without_manifest(tmp_path):
    with pytest.raises(ValueError, match="submission.json is missing"):
        submission_directories(tmp_path, [tmp_path])

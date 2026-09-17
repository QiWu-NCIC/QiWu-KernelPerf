from pathlib import Path

import pytest

from kernelperf.benchmark import benchmark_registry_from_config
from kernelperf.submissions import load_submission_artifacts, request_for_submission


SUBMISSION = Path("submissions/spmv/rocsparse_legacy")


@pytest.mark.parametrize("operator_id", ["spmv.csr.fp32", "spmv.csr.fp64"])
def test_legacy_rocsparse_submission(operator_id: str):
    artifacts = load_submission_artifacts(
        SUBMISSION,
        operator_id=operator_id,
        language="hip",
    )

    assert len(artifacts) == 1
    assert artifacts[0].name == "rocSPARSE Legacy CSRMV (No Analysis)"
    assert artifacts[0].metadata["configuration_id"] == "no-analysis"
    assert artifacts[0].metadata["candidate_group"] == "rocSPARSE-Legacy-CSRMV"


def test_legacy_rocsparse_uses_typed_csrmv_without_analysis():
    source = (SUBMISSION / "adapter.cu").read_text(encoding="utf-8")

    assert "rocsparse_scsrmv" in source
    assert "rocsparse_dcsrmv" in source
    assert "nullptr, context->device_x" in source


@pytest.mark.parametrize("operator_id", ["spmv.csr.fp32", "spmv.csr.fp64"])
def test_legacy_rocsparse_passes_submission_validation(operator_id: str):
    benchmark = benchmark_registry_from_config("config/benchmarks.json").get("spmv")
    request = request_for_submission(
        SUBMISSION,
        backend_id="Z100-gfx906",
        dataset_id="suitesparse_sample_100",
        operator_id=operator_id,
        language="hip",
    )

    benchmark.validate_submission(request)

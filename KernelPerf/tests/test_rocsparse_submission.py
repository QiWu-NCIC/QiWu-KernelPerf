from pathlib import Path

import pytest

from kernelperf.benchmark import benchmark_registry_from_config
from kernelperf.submissions import load_submission_artifacts, request_for_submission


SUBMISSION = Path("submissions/spmv/rocsparse")


@pytest.mark.parametrize("operator_id", ["spmv.csr.fp32", "spmv.csr.fp64"])
def test_rocsparse_exposes_all_csr_algorithms(operator_id: str):
    artifacts = load_submission_artifacts(
        SUBMISSION,
        operator_id=operator_id,
        language="hip",
    )

    assert [item.metadata["configuration_id"] for item in artifacts] == [
        "default",
        "adaptive",
        "stream",
        "lrb",
        "nnzsplit",
    ]
    assert [item.name for item in artifacts] == [
        "rocSPARSE CSR Default",
        "rocSPARSE CSR Adaptive",
        "rocSPARSE CSR Stream",
        "rocSPARSE CSR LRB",
        "rocSPARSE CSR NNZ Split",
    ]
    assert all(item.language == "hip" for item in artifacts)
    assert all(item.metadata["base_format"] == "csr" for item in artifacts)
    assert all(item.metadata["build_profile"] == "rocsparse-hip" for item in artifacts)


def test_rocsparse_rejects_cuda():
    with pytest.raises(ValueError, match="supports languages"):
        load_submission_artifacts(
            SUBMISSION,
            operator_id="spmv.csr.fp32",
            language="cuda",
        )


def test_rocsparse_stream_variant_uses_csr_stream_enum():
    assert "rocsparse_spmv_alg_csr_stream" in (
        SUBMISSION / "variants" / "rowsplit.cu"
    ).read_text(encoding="utf-8")


def test_rocsparse_adapter_uses_v1_staged_api():
    source = (SUBMISSION / "adapter.cu").read_text(encoding="utf-8")

    assert "rocsparse_spmv_stage_buffer_size" in source
    assert "rocsparse_spmv_stage_preprocess" in source
    assert "rocsparse_spmv_stage_compute" in source
    assert "ROCSPARSE_VERSION_MAJOR" not in source


@pytest.mark.parametrize("operator_id", ["spmv.csr.fp32", "spmv.csr.fp64"])
def test_rocsparse_artifacts_pass_submission_validation(operator_id: str):
    benchmark = benchmark_registry_from_config("config/benchmarks.json").get("spmv")

    request = request_for_submission(
        SUBMISSION,
        backend_id="BW1000-gfx936",
        dataset_id="suitesparse_sample_100",
        operator_id=operator_id,
        language="hip",
    )
    benchmark.validate_submission(request)

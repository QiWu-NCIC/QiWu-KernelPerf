from pathlib import Path

import pytest

from kernelperf.submissions import load_submission_artifacts


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
        "rowsplit",
        "lrb",
        "nnzsplit",
    ]
    assert [item.name for item in artifacts] == [
        "rocSPARSE upstream CSR Default",
        "rocSPARSE upstream CSR Adaptive",
        "rocSPARSE upstream CSR Rowsplit",
        "rocSPARSE upstream CSR LRB",
        "rocSPARSE upstream CSR NNZ Split",
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

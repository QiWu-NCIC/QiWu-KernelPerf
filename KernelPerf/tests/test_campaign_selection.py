from __future__ import annotations

from pathlib import Path

from kernelperf.campaign_selection import load_campaign_selection


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "config/campaigns/suitesparse_all.json"


def _selection(backend: str, benchmark: str) -> dict[str, set[str]]:
    return {
        path.name: set(configuration_ids)
        for path, configuration_ids in load_campaign_selection(SELECTION, backend, benchmark)
    }


def test_cuda_full_selection_matches_sample100_configuration_sets():
    expected_cusparse = {
        f"{format_name}-{algorithm}"
        for format_name in ("coo", "csc", "csr")
        for algorithm in ("alg1", "alg2", "default")
    } | {
        f"sell-c{c}-{algorithm}"
        for c in (1, 2, 4, 8, 16, 32, 64, 128)
        for algorithm in ("alg1", "default")
    } | {"sell-nrows-alg1", "sell-nrows-default"}
    expected_spmv = {
        "alphasparse": {"scalar", "vector", "adaptive", "merge", "line-enhance", "flat1", "flat4", "flat8"},
        "csr5": set(),
        "csr_adaptive": set(),
        "cusparse": expected_cusparse,
        "ghost_sell": {"sigma-32768"},
    }
    expected_h100_spmv = {**expected_spmv, "ghost_sell": {"sigma-131072"}}
    expected_spmm = {
        "alphasparse": {"csr-alg1", "csr-alg2", "csr-alg3", "csr-alg4", "csr-alg5"},
        "cusparse": {"csr-default", "csr-alg1", "csr-alg2", "csr-alg3"},
    }

    assert _selection("A100-SXM4-80GB", "spmv") == expected_spmv
    assert _selection("RTX5090-SL3061", "spmv") == expected_spmv | {"ghost_sell": {"sigma-32"}}
    assert _selection("H100-SXM5-80GB", "spmv") == expected_h100_spmv
    for backend in ("A100-SXM4-80GB", "H100-SXM5-80GB", "RTX5090-SL3061"):
        assert _selection(backend, "spmm") == expected_spmm


def test_cusparse_full_selection_has_exact_ordered_candidate_catalog():
    expected = [
        "coo-alg1", "coo-alg2", "coo-default",
        "csc-alg1", "csc-alg2", "csc-default",
        "csr-alg1", "csr-alg2", "csr-default",
        *[
            configuration
            for c in (1, 128, 16, 2, 32, 4, 64, 8)
            for configuration in (f"sell-c{c}-alg1", f"sell-c{c}-default")
        ],
        "sell-nrows-alg1", "sell-nrows-default",
    ]
    for backend in ("A100-SXM4-80GB", "H100-SXM5-80GB", "RTX5090-SL3061"):
        cusparse = next(
            configurations
            for submission, configurations in load_campaign_selection(SELECTION, backend, "spmv")
            if submission.name == "cusparse"
        )
        assert list(cusparse) == expected


def test_hip_full_selection_uses_only_dedicated_rocsparse_adapters():
    expected_spmv = {
        "alphasparse": {"scalar", "vector", "adaptive", "merge", "line-enhance", "flat1", "flat4", "flat8"},
        "rocsparse": {"adaptive", "stream", "lrb"},
    }
    expected_spmm = {
        "alphasparse": {"csr-alg1", "csr-alg2", "csr-alg3", "csr-alg4", "csr-alg5"},
        "rocsparse": {"csr", "csr-row-split", "csr-nnz-split", "csr-merge-path"},
    }

    for backend in ("BW1000-gfx936", "Z100-gfx906"):
        assert _selection(backend, "spmv") == expected_spmv
        assert _selection(backend, "spmm") == expected_spmm

from __future__ import annotations

from pathlib import Path

from kernelperf.campaign_selection import load_campaign_selection


SELECTION = Path("config/campaigns/suitesparse_all.json")


def _selection(backend: str, benchmark: str) -> dict[str, set[str]]:
    return {
        path.name: set(configuration_ids)
        for path, configuration_ids in load_campaign_selection(SELECTION, backend, benchmark)
    }


def test_cuda_full_selection_matches_sample100_configuration_sets():
    expected_spmv = {
        "alphasparse": {"scalar", "vector", "adaptive", "merge", "line-enhance", "flat1", "flat4", "flat8"},
        "csr5": set(),
        "csr_adaptive": set(),
        "cusparse": {"coo-default", "coo-alg1", "coo-alg2", "csr-default", "csr-alg1", "csr-alg2", "csc-default", "csc-alg1", "csc-alg2", "sell-c16-default", "sell-c16-alg1", "sell-nrows-default", "sell-nrows-alg1"},
        "ghost_sell": {"sigma-32768"},
    }
    expected_h100_spmv = {**expected_spmv, "cusparse": expected_spmv["cusparse"] - {"sell-c16-alg1"} | {"sell-c8-alg1"}, "ghost_sell": {"sigma-131072"}}
    expected_spmm = {
        "alphasparse": {"csr-alg1", "csr-alg2", "csr-alg3", "csr-alg4", "csr-alg5"},
        "cusparse": {"csr-default", "csr-alg1", "csr-alg2", "csr-alg3"},
    }

    assert _selection("A100-SXM4-80GB", "spmv") == expected_spmv
    assert _selection("RTX5090-SL3061", "spmv") == expected_spmv | {"ghost_sell": {"sigma-32"}}
    assert _selection("H100-SXM5-80GB", "spmv") == expected_h100_spmv
    for backend in ("A100-SXM4-80GB", "H100-SXM5-80GB", "RTX5090-SL3061"):
        assert _selection(backend, "spmm") == expected_spmm


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

from __future__ import annotations

import json
import urllib.error

from kernelperf import leaderboard as leaderboard_module
from kernelperf.leaderboard import GithubContentsClient, LeaderboardPublisher


def test_github_get_treats_missing_content_as_absent(monkeypatch):
    def missing(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.github.test/missing",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(leaderboard_module.urllib.request, "urlopen", missing)
    client = GithubContentsClient("owner/repository", "results", "token")

    assert client.get("public/data/new.csv") is None


def test_publisher_adds_an_incremental_csv_and_manifest_entry(monkeypatch):
    files: dict[str, bytes] = {}

    class MemoryClient:
        def __init__(self, repository, branch, token):
            assert repository == "owner/repository"
            assert branch == "results"
            assert token == "secret"

        def get(self, path):
            content = files.get(path)
            return None if content is None else (content, "sha")

        def put(self, path, content, message):
            files[path] = content
            return {"commit": {"html_url": "https://github.test/commit"}}

    monkeypatch.setenv("KERNELPERF_GITHUB_TOKEN", "secret")
    monkeypatch.setattr(leaderboard_module, "GithubContentsClient", MemoryClient)
    publisher = LeaderboardPublisher({
        "repository": "owner/repository",
        "branch": "results",
        "data_prefix": "public/data/results",
        "manifest_path": "public/data/index.json",
        "public_url": "https://github.test/leaderboard",
    })

    response = publisher.publish(
        "job-a100-spmv",
        {
            "submission_id": "job-a100-spmv",
            "method_id": "sell-c32",
            "method_name": "SELL C32",
            "base_format": "sell",
            "operator_id": "spmv.csr.fp32",
            "dtype": "fp32",
            "backend_id": "a100-server",
            "hardware": "NVIDIA A100",
            "peak_gflops": 19500,
            "dataset_id": "suitesparse_sample_100",
            "source_kind": "source",
            "created_at": "2026-08-14T00:00:00Z",
        },
        "schema_version,method_id\n2,sell-c32\n",
        {
            "entry_source": "adapter.cu",
            "compile_units": ["upstream/spmv.cu"],
            "files": [
                {"path": "adapter.cu", "content": "// adapter\n"},
                {"path": "upstream/spmv.cu", "content": "// upstream\n"},
            ],
        },
    )

    result_path = "public/data/results/spmv/a100-server/job-a100-spmv.csv"
    assert response["result_path"] == result_path
    assert result_path in files
    manifest = json.loads(files["public/data/index.json"])
    assert manifest["submissions"][0]["path"] == result_path.removeprefix("public/")
    assert manifest["submissions"][0]["base_format"] == "sell"
    source_root = "public/source/job-a100-spmv"
    assert files[f"{source_root}/files/adapter.cu"] == b"// adapter\n"
    assert files[f"{source_root}/files/upstream/spmv.cu"] == b"// upstream\n"
    source_manifest = json.loads(files[f"{source_root}/plugin.json"])
    assert source_manifest["entry_source"] == "adapter.cu"
    assert manifest["submissions"][0]["source_manifest"] == (
        "source/job-a100-spmv/plugin.json"
    )

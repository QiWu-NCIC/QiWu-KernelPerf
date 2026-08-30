from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


RESULT_COLUMNS = [
    "schema_version", "submission_id", "method_id", "method_name", "configuration_id",
    "candidate_group", "selection_role", "selected_from", "base_format",
    "operator_id", "dtype", "backend_id", "hardware", "peak_gflops", "job_id",
    "dataset_id", "matrix_id", "matrix_name", "rows", "cols", "nnz", "status", "operations",
    "preprocess_ms", "solve_ms", "solve_gflops", "solve_only_efficiency_percent",
    "timestamp", "source_kind",
]


def slug(value: str, *, fallback: str = "submission") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or fallback)[:96]


def make_submission(
    *,
    job: Any,
    results: list[dict[str, Any]],
    backend: Any,
    operator: Any,
    kernel: Any,
    method_id: str | None = None,
    method_name: str | None = None,
    configuration_id: str | None = None,
    candidate_group: str | None = None,
    selection_role: str | None = None,
    selected_from: str = "",
    source_kind: str | None = None,
    base_format: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    if not results:
        raise ValueError("cannot publish an empty result set")
    kernel_metadata = kernel.metadata
    method_id = method_id or slug(kernel.name, fallback=job.generator_id)
    method_name = method_name or kernel.name
    configuration_id = configuration_id or str(kernel_metadata.get("configuration_id", ""))
    candidate_group = candidate_group or str(kernel_metadata.get("candidate_group", ""))
    selection_role = selection_role or str(kernel_metadata.get("selection_role", "candidate"))
    source_kind = source_kind or kernel.kind
    identity = configuration_id or method_id
    submission_id = slug(f"{job.job_id}-{backend.backend_id}-{operator.op_id}-{identity}")
    base_format = base_format or str(kernel.metadata["base_format"])
    peak_gflops = float(backend.peak_gflops_by_dtype.get(operator.dtype, backend.peak_gflops))
    if peak_gflops <= 0:
        raise ValueError(f"no positive {operator.dtype} peak configured for {backend.backend_id}")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in results:
        metadata = result.get("metadata") or {}
        operations = float(metadata.get("operations", 2 * int(result["nnz"])))
        solve_ms = float(result["runtime_ms"])
        solve_gflops = float(result["gflops"])
        efficiency = (
            solve_gflops / peak_gflops * 100
            if result["status"] == "pass" and solve_gflops > 0
            else 0.0
        )
        writer.writerow({
            "schema_version": 2,
            "submission_id": submission_id,
            "method_id": method_id,
            "method_name": method_name,
            "configuration_id": configuration_id,
            "candidate_group": candidate_group,
            "selection_role": selection_role,
            "selected_from": selected_from,
            "base_format": base_format,
            "operator_id": operator.op_id,
            "dtype": operator.dtype,
            "backend_id": backend.backend_id,
            "hardware": backend.name,
            "peak_gflops": peak_gflops,
            "job_id": job.job_id,
            "dataset_id": job.dataset_id or "none",
            "matrix_id": result["matrix_id"],
            "matrix_name": result["matrix_name"],
            "rows": result["rows"],
            "cols": result["cols"],
            "nnz": result["nnz"],
            "status": result["status"],
            "operations": operations,
            "preprocess_ms": result["preprocess_ms"],
            "solve_ms": solve_ms,
            "solve_gflops": solve_gflops,
            "solve_only_efficiency_percent": efficiency,
            "timestamp": result["timestamp"],
            "source_kind": source_kind,
        })

    entry = {
        "submission_id": submission_id,
        "method_id": method_id,
        "method_name": method_name,
        "configuration_id": configuration_id,
        "candidate_group": candidate_group,
        "selection_role": selection_role,
        "selected_from": selected_from,
        "base_format": base_format,
        "operator_id": operator.op_id,
        "dtype": operator.dtype,
        "backend_id": backend.backend_id,
        "hardware": backend.name,
        "peak_gflops": peak_gflops,
        "dataset_id": job.dataset_id or "none",
        "source_kind": source_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return submission_id, entry, output.getvalue()


class GithubContentsClient:
    def __init__(self, repository: str, branch: str, token: str) -> None:
        self.repository = repository
        self.branch = branch
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repository}/contents"

    def get(self, path: str) -> tuple[bytes, str] | None:
        encoded_path = urllib.parse.quote(path, safe="/")
        query = urllib.parse.urlencode({"ref": self.branch})
        try:
            payload = self._request("GET", f"{self.base_url}/{encoded_path}?{query}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return base64.b64decode(payload["content"]), str(payload["sha"])

    def put(self, path: str, content: bytes, message: str) -> dict[str, Any]:
        current = self.get(path)
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content).decode(),
            "branch": self.branch,
        }
        if current is not None:
            payload["sha"] = current[1]
        encoded_path = urllib.parse.quote(path, safe="/")
        return self._request("PUT", f"{self.base_url}/{encoded_path}", payload)

    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "KernelPerf",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API {exc.code}: {detail[-1000:]}") from exc


class LeaderboardPublisher:
    def __init__(self, config: dict[str, str]) -> None:
        self.config = config
        self.token = os.environ.get("KERNELPERF_GITHUB_TOKEN", "")
        self._manifest_lock = threading.Lock()

    @property
    def public_url(self) -> str:
        return self.config.get("public_url", "")

    @property
    def branch_url(self) -> str:
        repository = self.config.get("repository", "")
        branch = self.config.get("branch", "")
        return f"https://github.com/{repository}/tree/{branch}" if repository and branch else ""

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.config.get("repository") and self.config.get("branch"))

    def publish(
        self,
        submission_id: str,
        entry: dict[str, Any],
        csv_content: str,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("GitHub publishing requires KERNELPERF_GITHUB_TOKEN")
        client = GithubContentsClient(
            self.config["repository"],
            self.config["branch"],
            self.token,
        )
        prefix = self.config.get("data_prefix", "public/data/results").rstrip("/")
        result_path = f"{prefix}/spmv/{entry['backend_id']}/{submission_id}.csv"
        entry = {**entry, "path": result_path.removeprefix("public/")}
        csv_commit = client.put(
            result_path,
            csv_content.encode(),
            f"Add KernelPerf result {submission_id}",
        )

        source_paths: list[str] = []
        if source is not None:
            source_prefix = self.config.get("source_prefix", "public/source").rstrip("/")
            source_id = str(source.get("source_sha256") or submission_id)
            source_root = f"{source_prefix}/{source_id}"
            for item in source["files"]:
                source_path = f"{source_root}/files/{item['path']}"
                client.put(
                    source_path,
                    str(item["content"]).encode(),
                    f"Add KernelPerf source {submission_id}: {item['path']}",
                )
                source_paths.append(source_path)
            source_manifest_path = f"{source_root}/plugin.json"
            source_manifest = {
                "submission_id": submission_id,
                **source.get("plugin", {}),
            }
            source_manifest.setdefault("entry_source", source["entry_source"])
            source_manifest.setdefault("compile_units", source["compile_units"])
            source_manifest.setdefault("files", [item["path"] for item in source["files"]])
            if source.get("source_sha256"):
                source_manifest["source_sha256"] = source["source_sha256"]
            client.put(
                source_manifest_path,
                (json.dumps(source_manifest, indent=2) + "\n").encode(),
                f"Index KernelPerf source {submission_id}",
            )
            source_paths.append(source_manifest_path)
            entry = {
                **entry,
                "source_manifest": source_manifest_path.removeprefix("public/"),
            }
            if source.get("source_sha256"):
                entry["source_sha256"] = source["source_sha256"]

        manifest_path = self.config.get("manifest_path", "public/data/index.json")
        manifest_commit = self._update_manifest(
            client,
            manifest_path,
            submission_id,
            entry,
        )
        return {
            "published": True,
            "submission_id": submission_id,
            "result_path": result_path,
            "leaderboard_url": self.public_url,
            "branch_url": self.branch_url,
            "csv_commit_url": csv_commit.get("commit", {}).get("html_url"),
            "manifest_commit_url": manifest_commit.get("commit", {}).get("html_url"),
            "source_paths": source_paths,
        }

    def _update_manifest(
        self,
        client: GithubContentsClient,
        manifest_path: str,
        submission_id: str,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        with self._manifest_lock:
            for attempt in range(3):
                current = client.get(manifest_path)
                manifest = (
                    json.loads(current[0])
                    if current is not None
                    else {
                        "schema_version": 2,
                        "result_schema": "kernelperf-spmv-v2",
                        "baselines": [],
                        "submissions": [],
                    }
                )
                manifest.setdefault("baselines", [])
                manifest["submissions"] = [
                    item for item in manifest.get("submissions", [])
                    if item.get("submission_id") != submission_id
                ]
                manifest["submissions"].append(entry)
                manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    return client.put(
                        manifest_path,
                        (json.dumps(manifest, indent=2) + "\n").encode(),
                        f"Index KernelPerf result {submission_id}",
                    )
                except RuntimeError as exc:
                    if "GitHub API 409" not in str(exc) or attempt == 2:
                        raise
            raise RuntimeError("GitHub manifest update retry exhausted")

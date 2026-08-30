from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: str) -> dict[str, Any]:
    selected = Path(path)
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    return json.loads(selected.read_text())


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    admin: bool = False,
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"} if data else {}
    if admin:
        token = os.environ.get("KERNELPERF_ADMIN_TOKEN")
        if not token:
            raise SystemExit("KERNELPERF_ADMIN_TOKEN is required")
        headers["x-kernelperf-admin-token"] = token
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method or ("POST" if data else "GET"),
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from exc


def wait_for_job(api: str, job_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = request_json(f"{api}/api/v1/jobs/{job_id}")
        print(f"{job_id}: {job['status']}")
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(2)
    raise SystemExit(f"timed out waiting for {job_id}")

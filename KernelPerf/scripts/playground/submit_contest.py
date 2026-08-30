from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from spmv_artifact import add_artifact_arguments, artifact_from_args


def request_json(
    url: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a custom-format SpMV implementation to a contest."
    )
    parser.add_argument("--api", default="http://10.18.96.188:8080")
    parser.add_argument("--contest-id", default="TEST")
    parser.add_argument("--nick-name", required=True)
    parser.add_argument(
        "--operator",
        choices=("spmv.csr.fp32", "spmv.csr.fp64"),
        default="spmv.csr.fp32",
    )
    add_artifact_arguments(parser)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    artifact = artifact_from_args(args, parser)
    payload = {
        "contest-id": args.contest_id,
        "nick-name": args.nick_name,
        "operator_ids": [args.operator],
        "kernels": [artifact],
    }
    api = args.api.rstrip("/")
    submitted = request_json(f"{api}/api/v1/jobs", payload)
    print(json.dumps(submitted, indent=2))
    if not args.wait:
        return

    job_id = str(submitted["job_id"])
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        job = request_json(f"{api}/api/v1/jobs/{job_id}")
        print(f"{job_id}: {job['status']}")
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            if job["status"] != "succeeded":
                raise SystemExit(json.dumps(job, indent=2))
            print(f"contest leaderboard: {api}/contest/{args.contest_id}")
            return
        time.sleep(2)
    raise SystemExit(f"timed out waiting for {job_id}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_json, request_json, wait_for_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one configured KernelPerf task.")
    parser.add_argument("--deployment", default="config/deployment.json")
    parser.add_argument("--payload", required=True, help="Job request JSON file")
    parser.add_argument("--admin", action="store_true", help="Attach the administrator token")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    deployment = load_json(args.deployment)
    api = str(deployment["entry"]["public_url"]).rstrip("/")
    payload = json.loads(Path(args.payload).read_text())
    submitted = request_json(
        f"{api}/api/v1/jobs", payload, admin=args.admin
    )
    print(json.dumps(submitted, ensure_ascii=False, indent=2))
    if args.wait:
        job = wait_for_job(api, str(submitted["job_id"]), args.timeout)
        if job["status"] != "succeeded":
            raise SystemExit(json.dumps(job, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

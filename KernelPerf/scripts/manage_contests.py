from __future__ import annotations

import argparse
import json
from datetime import datetime

from common import load_json, request_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage KernelPerf contests through the REST API.")
    parser.add_argument("--deployment", default="config/deployment.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    delete = commands.add_parser("delete")
    delete.add_argument("--contest-id", required=True)
    delete.add_argument("--yes", action="store_true")
    add = commands.add_parser("add")
    add.add_argument("--contest-id", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--start-time", default=datetime.now().astimezone().isoformat())
    add.add_argument("--backend-id", required=True)
    add.add_argument("--dataset-id", required=True)
    add.add_argument("--benchmark", required=True)
    selection = add.add_mutually_exclusive_group(required=True)
    selection.add_argument("--operator-id", action="append")
    selection.add_argument("--all-operators", action="store_true")
    args = parser.parse_args()

    api = str(load_json(args.deployment)["entry"]["public_url"]).rstrip("/")
    if args.command == "list":
        result = request_json(f"{api}/api/v1/contests")
    elif args.command == "delete":
        if not args.yes:
            parser.error("--yes is required for deletion")
        result = request_json(
            f"{api}/api/v1/contests/{args.contest_id}", method="DELETE", admin=True
        )
    else:
        payload = {
            "contest_id": args.contest_id,
            "name": args.name,
            "start_time": args.start_time,
            "backend_id": args.backend_id,
            "dataset_id": args.dataset_id,
            "suite": args.benchmark,
        }
        if args.all_operators:
            payload["all_operators"] = True
        elif len(args.operator_id) == 1:
            payload["operator_id"] = args.operator_id[0]
        else:
            payload["operator_ids"] = args.operator_id
        result = request_json(f"{api}/api/v1/contests", payload, admin=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from spmv_artifact import add_artifact_arguments, artifact_from_args


SUITE_OPERATORS = {
    "spmv": ("spmv.csr.fp32", "spmv.csr.fp64"),
}


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
        description="Submit a custom-format SpMV implementation."
    )
    parser.add_argument("--api", default="http://10.18.96.188:8080")
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--suite", choices=tuple(SUITE_OPERATORS), default="spmv")
    parser.add_argument("--dataset-id", default="suitesparse_sample_100")
    parser.add_argument("--matrix-id", action="append", default=[])
    parser.add_argument("--operator", choices=tuple(value for values in SUITE_OPERATORS.values() for value in values), default=None)
    parser.add_argument("--contest-id", default=None)
    parser.add_argument("--nick-name", default=None)
    add_artifact_arguments(parser)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish each successful backend result to the GitHub leaderboard.",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--sweep-manifest",
        type=Path,
        default=None,
        help="JSON list of configurations; each item has entry_source, configuration_id, and optional method_name.",
    )
    args = parser.parse_args()

    if args.operator is None:
        args.operator = SUITE_OPERATORS[args.suite][0]
    if args.operator not in SUITE_OPERATORS[args.suite]:
        parser.error(f"{args.operator} is not an operator in suite {args.suite}")

    if args.nick_name and not args.contest_id:
        parser.error("--nick-name requires --contest-id")
    if args.publish and (not args.wait or args.contest_id):
        parser.error("--publish requires --wait and is not used for contest jobs")
    if args.object and not args.contest_id and len(args.backend) != 1:
        parser.error("object submissions require exactly one --backend")
    if args.sweep_manifest and args.object:
        parser.error("--sweep-manifest is supported for source trees only")

    artifacts = []
    if args.sweep_manifest:
        if not args.source_dir:
            parser.error("--sweep-manifest requires --source-dir")
        try:
            sweep = json.loads(args.sweep_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --sweep-manifest: {exc}")
        if not isinstance(sweep, list) or not sweep:
            parser.error("--sweep-manifest must contain a non-empty JSON list")
        for item in sweep:
            if not isinstance(item, dict) or not item.get("entry_source") or not item.get("configuration_id"):
                parser.error("each sweep item requires entry_source and configuration_id")
            metadata = {
                "configuration_id": str(item["configuration_id"]),
                "candidate_group": str(item.get("candidate_group") or args.candidate_group or args.method_name or "sweep"),
            }
            if item.get("base_format"):
                metadata["base_format"] = str(item["base_format"])
            artifacts.append(
                artifact_from_args(
                    args,
                    parser,
                    entry_source_override=str(item["entry_source"]),
                    method_name_override=str(item.get("method_name") or item["configuration_id"]),
                    metadata_overrides=metadata,
                )
            )
    else:
        artifacts.append(artifact_from_args(args, parser))
    selected_backends = args.backend or ["rtx5090-workstation", "A100-SXM4-80GB"]
    payload: dict[str, object] = {
        "generator_id": str(args.candidate_group or artifacts[0]["name"]),
        "submitter": "kernel-client",
        "operator_ids": [args.operator],
        "kernels": artifacts,
        "priority": 50,
        "tags": {
            "benchmark": args.suite,
            "base_format": args.base_format,
            "artifact_kind": str(artifacts[0]["kind"]),
        },
    }
    if args.candidate_group:
        payload["tags"]["candidate_group"] = args.candidate_group
    if args.contest_id:
        payload["contest-id"] = args.contest_id
        if args.nick_name:
            payload["nick-name"] = args.nick_name
    else:
        payload.update(
            {
                "backends": selected_backends,
                "suites": [args.suite],
                "dataset_id": args.dataset_id,
                "matrix_ids": args.matrix_id or None,
            }
        )
        payload["tags"]["dataset"] = args.dataset_id

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
            break
        time.sleep(2)
    else:
        raise SystemExit(f"timed out waiting for {job_id}")

    if args.contest_id:
        print(f"contest leaderboard: {api}/contest/{args.contest_id}")
        return

    configuration_ids = [
        str(artifact.get("metadata", {}).get("configuration_id", ""))
        for artifact in artifacts
    ]
    if args.sweep_manifest:
        configuration_ids.append("per-matrix-best")
    configuration_ids = [value for value in configuration_ids if value]
    if not configuration_ids:
        configuration_ids = [args.configuration_id or ""]
    for backend_id in selected_backends:
        for configuration_id in configuration_ids:
            selection = {
                "job_id": job_id,
                "backend_id": backend_id,
                "suite": args.suite,
                "operator_id": args.operator,
            }
            if configuration_id:
                selection["configuration_id"] = configuration_id
            if configuration_id == "per-matrix-best" and args.candidate_group:
                selection["candidate_group"] = args.candidate_group
            print(
                "result CSV: "
                + api
                + "/api/v1/results.csv?"
                + urllib.parse.urlencode(selection)
            )
            if args.publish:
                published = request_json(
                    f"{api}/api/v1/leaderboard/submissions",
                    selection,
                )
                print(json.dumps(published, indent=2))


if __name__ == "__main__":
    main()

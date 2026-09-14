from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def submission_directories(root: Path, selected: list[Path]) -> list[Path]:
    if selected:
        directories = [path.resolve() for path in selected]
    else:
        directories = sorted(path.parent for path in root.rglob("submission.json"))
    missing = [path for path in directories if not (path / "submission.json").is_file()]
    if missing:
        raise ValueError("submission.json is missing from: " + ", ".join(map(str, missing)))
    return directories


def supported_operators(submission: Path) -> list[str]:
    manifest = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    declared = str(manifest.get("operator_id", "")).strip()
    raw_operators = manifest.get("supported_operators")
    if not isinstance(raw_operators, list):
        raw_operators = [declared]
    operators = [str(value).strip() for value in raw_operators]
    return list(dict.fromkeys(value for value in operators if value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reviewed submissions sequentially on one configured backend"
    )
    parser.add_argument("--config", default="config/service.json")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--submission-root", type=Path, default=Path("submissions"))
    parser.add_argument("--submission", type=Path, action="append", default=[])
    parser.add_argument(
        "--operator",
        action="append",
        default=[],
        help="Operator ID to evaluate; repeat as needed. The default is every declared operator.",
    )
    parser.add_argument("--matrix-id", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    try:
        submissions = submission_directories(args.submission_root, args.submission)
    except ValueError as exc:
        parser.error(str(exc))
    requested = set(args.operator)
    failures: list[str] = []
    evaluated = 0
    for submission in submissions:
        operators = supported_operators(submission)
        if requested:
            operators = [operator for operator in operators if operator in requested]
        for operator in operators:
            evaluated += 1
            command = [
                sys.executable,
                "-m",
                "kernelperf.cli",
                "evaluate",
                "--config",
                args.config,
                "--submission",
                str(submission),
                "--backend",
                args.backend,
                "--dataset-id",
                args.dataset_id,
                "--operator",
                operator,
                "--timeout",
                str(args.timeout),
            ]
            for matrix_id in args.matrix_id:
                command.extend(["--matrix-id", matrix_id])
            print(f"\n=== {submission.name}: {operator} ===", flush=True)
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                label = f"{submission}:{operator}"
                failures.append(label)
                if args.fail_fast:
                    raise SystemExit(completed.returncode)

    if not evaluated:
        parser.error("no submission/operator combinations matched")
    if failures:
        print("\nFailed evaluations:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nCompleted {evaluated} submission/operator evaluations.")


if __name__ == "__main__":
    main()

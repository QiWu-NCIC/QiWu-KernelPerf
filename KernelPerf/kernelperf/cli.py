from __future__ import annotations

import argparse
import time
from pathlib import Path

from .models import JobStatus
from .runtime import create_runtime
from .submissions import request_for_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelPerf maintainer tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a reviewed submission without exposing a public API"
    )
    evaluate.add_argument("--config", default="config/service.json")
    evaluate.add_argument("--submission", type=Path, required=True)
    evaluate.add_argument("--backend", required=True)
    evaluate.add_argument("--dataset-id", required=True)
    evaluate.add_argument("--operator", default=None)
    evaluate.add_argument("--matrix-id", action="append", default=[])
    evaluate.add_argument("--configuration-id", action="append", default=[])
    evaluate.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    runtime = create_runtime(args.config)
    backend = runtime.backends.get(args.backend)
    cuda_version = str(
        backend.labels.get("cuda_version")
        or backend.spec.get("environment", {}).get("variables", {}).get("KERNELPERF_CUDA_VERSION", "")
    )
    request = request_for_submission(
        args.submission,
        backend_id=args.backend,
        dataset_id=args.dataset_id,
        operator_id=args.operator,
        matrix_ids=args.matrix_id,
        configuration_ids=args.configuration_id,
        cuda_version=cuda_version,
    )
    scheduler = runtime.scheduler
    scheduler.start()
    try:
        job, position = scheduler.submit(request)
        print(f"accepted job={job.job_id} queue_position={position}")
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            current = scheduler.get_job(job.job_id)
            if current is None:
                raise SystemExit("job disappeared from scheduler")
            print(f"{current.job_id}: {current.status.value}")
            if current.status in {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}:
                if current.status != JobStatus.succeeded:
                    raise SystemExit(current.error or current.status.value)
                return
            time.sleep(2)
        raise SystemExit(f"timed out waiting for {job.job_id}")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()

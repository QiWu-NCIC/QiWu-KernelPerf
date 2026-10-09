from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelperf.models import JobRecord, JobStatus, utc_now_iso
from kernelperf.campaign_selection import load_campaign_selection
from kernelperf.runtime import create_runtime
from kernelperf.submissions import load_submission_artifacts


MAIN_RHS = (2, 4, 8, 16, 32, 64, 128)


def row_key(row):
    metadata = row.get("metadata") or {}
    implementation = metadata.get("implementation") or {}
    candidate_group = implementation.get("candidate_group") or ""
    configuration_id = implementation.get("configuration_id") or row["kernel_name"]
    return candidate_group, configuration_id, row["matrix_id"], int(metadata.get("rhs_columns", 0))


def kernel_key(kernel):
    group = str(kernel.metadata.get("candidate_group") or "")
    configuration = str(kernel.metadata.get("configuration_id") or kernel.name)
    return group, configuration


def completed_keys(rows, *, retry_failures=False):
    return {
        row_key(row) for row in rows
        if not retry_failures or row.get("status") == "pass"
    }


def phase_cases(matrices, phase, rhs_columns=MAIN_RHS):
    smoke = matrices[:3]
    if phase == "smoke":
        return [(matrix, [1, *rhs_columns]) for matrix in smoke]
    if phase == "n2" and 2 in rhs_columns:
        return [(matrix, [2]) for matrix in matrices]
    if phase == "n2":
        return []
    return [(matrix, [1, *rhs_columns]) for matrix in matrices]


def fingerprint(benchmark, backend, kernels, operator, matrices):
    digest = hashlib.sha256()
    data = {
        "kernels": [kernel.model_dump() for kernel in kernels],
        "operator": operator.model_dump(),
        "backend": backend.spec,
        "matrices": [matrix.model_dump() for matrix in matrices],
    }
    digest.update(json.dumps(data, sort_keys=True).encode())
    for filename in [benchmark.template_path, benchmark.contract_path,
                     benchmark.contract_path.with_name("gpu_runtime.h"),
                     Path(__file__).resolve().parents[1] / "benchmarks/spmm/driver.py"]:
        digest.update(Path(filename).read_bytes())
    return digest.hexdigest()


def persist_checkpoint(runtime, job, complete=False):
    job.status = JobStatus.succeeded if complete else JobStatus.running
    if complete:
        job.finished_at = utc_now_iso()
    runtime.db.upsert_job(job)
    export_job = job.model_copy(deep=False, update={"status": JobStatus.succeeded})
    runtime.exporter.export_job(export_job)


def run_campaign(runtime, args):
    backend = runtime.backends.get(args.backend)
    if backend.spec.get("transport") != "local":
        raise ValueError("run the campaign on the allocated compute node with a local worker")
    benchmark = runtime.benchmarks.get("spmm")
    _, all_matrices = runtime.datasets.get(args.dataset_id)
    all_shard_matrices = all_matrices[args.shard_index :: args.shard_count]
    matrices = [matrix for matrix in all_shard_matrices
                if matrix.local_path and Path(matrix.local_path).is_file()]
    missing_count = len(all_shard_matrices) - len(matrices)
    if missing_count:
        print(json.dumps({"status": "partial_dataset", "missing_matrices": missing_count,
                          "shard_index": args.shard_index}), flush=True)
    if not matrices:
        print(json.dumps({"status": "waiting_for_matrix_shard",
                          "shard_index": args.shard_index}), flush=True)
        return 75
    rhs_columns = tuple(sorted(set(args.rhs_columns or MAIN_RHS)))
    invalid_rhs = set(rhs_columns) - set(MAIN_RHS)
    if invalid_rhs:
        raise ValueError(f"unsupported SpMM RHS columns: {sorted(invalid_rhs)}")
    healthy, message = backend.healthcheck()
    if not healthy:
        raise RuntimeError(message)
    language = "hip" if "hip" in backend.info().supported_languages and "cuda" not in backend.info().supported_languages else "cuda"
    if args.selection_file:
        method_specs = load_campaign_selection(Path(args.selection_file), args.backend, "spmm")
    else:
        methods = args.submission or [Path("submissions/spmm/alphasparse"), Path("submissions/spmm/rocsparse" if language == "hip" else "submissions/spmm/cusparse")]
        method_specs = [(path, tuple(args.configuration_id)) for path in methods]
    requested_operators = set(args.operator or [item.op_id for item in benchmark.operators()])
    operators = [item for item in benchmark.operators() if item.op_id in requested_operators]
    unknown_operators = requested_operators - {item.op_id for item in operators}
    if unknown_operators:
        raise ValueError(f"unknown SpMM operators: {sorted(unknown_operators)}")
    jobs = []
    for method, configuration_ids in method_specs:
        for operator in operators:
            kernels = load_submission_artifacts(method, operator_id=operator.op_id,
                configuration_ids=list(configuration_ids) or None, language=language,
                cuda_version=str(backend.labels.get("cuda_version", "")))
            job_id = str(uuid5(NAMESPACE_URL, "|".join([args.campaign_id, args.backend,
                args.dataset_id, str(method.resolve()), operator.op_id,
                str(args.shard_index)])))
            expected = fingerprint(benchmark, backend, kernels, operator, all_shard_matrices)
            job = runtime.db.get_job(job_id)
            if job is None:
                job = JobRecord(job_id=job_id, generator_id="spmm-campaign", backends=[args.backend],
                    suites=["spmm"], dataset_id=args.dataset_id, kernels=kernels,
                    operator_ids=[operator.op_id], tags={"campaign_id": args.campaign_id, "fingerprint": expected},
                    started_at=utc_now_iso())
                runtime.db.upsert_job(job)
            elif job.tags.get("fingerprint") != expected:
                if not args.adopt_fingerprint:
                    raise ValueError("source, protocol, dataset, or worker changed; use a new campaign ID")
                job.tags["fingerprint"] = expected
                job.kernels = kernels
                runtime.db.upsert_job(job)
            jobs.append((job, operator))
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else float("inf")
    stop = []
    def request_stop(signum, frame):
        stop.append(signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    for phase in ("smoke", "n2", "full"):
        if args.phase != "all" and args.phase != phase:
            continue
        completion_phase = "full" if args.phase == "all" else args.phase
        for job, operator in jobs:
            rows = runtime.db.query_results(job_ids=[job.job_id])
            completed = completed_keys(rows, retry_failures=args.retry_failures)
            for kernel in job.kernels:
                for matrix, requested_rhs in phase_cases(matrices, phase, rhs_columns):
                    missing = [
                        rhs for rhs in requested_rhs
                        if (*kernel_key(kernel), matrix.matrix_id, rhs) not in completed
                    ]
                    if not missing:
                        continue
                    if stop or time.monotonic() >= deadline:
                        persist_checkpoint(runtime, job)
                        print(json.dumps({"status": "checkpointed", "phase": phase, "job_id": job.job_id}), flush=True)
                        return 75
                    scoped = operator.model_copy(deep=True)
                    scoped.metadata["driver_config"]["rhs_columns"] = [rhs for rhs in missing if rhs != 1]
                    scoped.metadata["driver_config"]["sanity_rhs_columns"] = [1] if 1 in missing else []
                    results = benchmark.run_case(backend, job, scoped, matrix, kernel)
                    for result in results:
                        result.metadata["campaign_phase"] = phase
                        result.metadata["source_fingerprint"] = job.tags["fingerprint"]
                        runtime.db.insert_result(result)
                        completed.add(row_key(result.model_dump()))
                    print(json.dumps({"phase": phase, "dtype": operator.dtype,
                        "method": kernel.name, "matrix": matrix.matrix_id,
                        "rhs": missing, "statuses": [result.status.value for result in results]}), flush=True)
                persist_checkpoint(runtime, job)
            target_keys = {
                (*kernel_key(kernel), matrix.matrix_id, rhs)
                for kernel in job.kernels
                for matrix, requested_rhs in phase_cases(matrices, completion_phase, rhs_columns)
                for rhs in requested_rhs
            }
            covered_keys = {
                row_key(row)
                for row in runtime.db.query_results(job_ids=[job.job_id])
            }
            persist_checkpoint(runtime, job, complete=target_keys.issubset(covered_keys))
    print(json.dumps({"status": "complete", "phase": args.phase,
        "jobs": [job.job_id for job, _ in jobs]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Checkpointed CSR SpMM campaign; execute inside tmux or a Slurm allocation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset-id", default="suitesparse_sample_100")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--submission", action="append", type=Path, default=[])
    parser.add_argument("--configuration-id", action="append", default=[])
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--phase", choices=["all", "smoke", "n2", "full"], default="all")
    parser.add_argument("--rhs-columns", nargs="+", type=int, default=None)
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--retry-failures", action="store_true",
                        help="retry existing fail/error rows once while preserving every attempt")
    parser.add_argument("--adopt-fingerprint", action="store_true",
                        help="adopt the current source and dataset fingerprint for this named campaign")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be within shard-count")
    runtime = create_runtime(args.config)
    lock = runtime.db.path.with_suffix(".campaign.lock")
    with lock.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_campaign(runtime, args)


if __name__ == "__main__":
    raise SystemExit(main())

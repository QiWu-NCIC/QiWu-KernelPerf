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

from kernelperf.benchmark import failure_metadata
from kernelperf.campaign_selection import load_campaign_selection
from kernelperf.models import BenchmarkResult, CaseStatus, JobRecord, JobStatus, utc_now_iso
from kernelperf.runtime import create_runtime
from kernelperf.submissions import load_submission_artifacts


def row_key(row):
    metadata = row.get("metadata") or {}
    implementation = metadata.get("implementation") or {}
    return (
        implementation.get("candidate_group") or "",
        implementation.get("configuration_id") or row["kernel_name"],
        row["matrix_id"],
    )


def kernel_key(kernel):
    metadata = kernel.metadata
    return str(metadata.get("candidate_group") or ""), str(
        metadata.get("configuration_id") or kernel.name
    )


def completed_keys(rows, retry_failures=False):
    return {
        row_key(row)
        for row in rows
        if not retry_failures or row.get("status") == "pass"
    }


def fingerprint(benchmark, backend, kernels, operator, matrices):
    digest = hashlib.sha256()
    payload = {
        "kernels": [kernel.model_dump() for kernel in kernels],
        "operator": operator.model_dump(),
        "backend": backend.spec,
        "matrices": [matrix.model_dump() for matrix in matrices],
    }
    digest.update(json.dumps(payload, sort_keys=True).encode())
    for filename in [
        benchmark.template_path,
        benchmark.contract_path,
        benchmark.contract_path.with_name("gpu_runtime.h"),
        Path(__file__).resolve().parents[1] / "benchmarks/spmv/driver.py",
    ]:
        digest.update(Path(filename).read_bytes())
    return digest.hexdigest()


def persist_checkpoint(runtime, job, complete=False):
    job.status = JobStatus.succeeded if complete else JobStatus.running
    if complete:
        job.finished_at = utc_now_iso()
    runtime.db.upsert_job(job)
    runtime.exporter.export_job(job.model_copy(deep=False, update={"status": JobStatus.succeeded}))


def failed_result(backend, benchmark, job, operator, matrix, kernel, error):
    info = backend.info()
    return BenchmarkResult(
        job_id=job.job_id,
        generator_id=job.generator_id,
        backend_id=info.backend_id,
        backend_kind=info.kind,
        suite=benchmark.benchmark_id,
        operator_id=operator.op_id,
        operator_name=operator.name,
        matrix_id=matrix.matrix_id,
        matrix_name=matrix.name,
        rows=matrix.rows,
        cols=matrix.cols,
        nnz=matrix.nnz,
        kernel_name=kernel.name,
        status=CaseStatus.failed,
        runtime_ms=0.0,
        gflops=0.0,
        arithmetic_intensity=0.0,
        metadata={
            "mode": "failed",
            "implementation": {
                "kind": kernel.kind,
                "base_format": kernel.metadata.get("base_format", "unknown"),
                "candidate_group": kernel.metadata.get("candidate_group", ""),
                "configuration_id": kernel.metadata.get("configuration_id", kernel.name),
            },
            **failure_metadata(error),
            "validation": {
                "method": "standard-program-comparison",
                "failure": "case did not complete",
            },
        },
    )


def run_campaign(runtime, args):
    backend = runtime.backends.get(args.backend)
    if backend.spec.get("transport") != "local":
        raise ValueError("run the campaign on the allocated compute node with a local worker")
    benchmark = runtime.benchmarks.get("spmv")
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
    healthy, message = backend.healthcheck()
    if not healthy:
        raise RuntimeError(message)
    language = (
        "hip"
        if "hip" in backend.info().supported_languages
        and "cuda" not in backend.info().supported_languages
        else "cuda"
    )
    if args.selection_file:
        method_specs = load_campaign_selection(Path(args.selection_file), args.backend, "spmv")
    else:
        methods = args.submission or sorted(Path("submissions/spmv").glob("*/submission.json"))
        methods = [path.parent if path.name == "submission.json" else path for path in methods]
        method_specs = [(path, tuple(args.configuration_id)) for path in methods]
    requested_operators = set(args.operator or [item.op_id for item in benchmark.operators()])
    operators = [item for item in benchmark.operators() if item.op_id in requested_operators]
    unknown_operators = requested_operators - {item.op_id for item in operators}
    if unknown_operators:
        raise ValueError(f"unknown SpMV operators: {sorted(unknown_operators)}")
    jobs = []
    for method, configuration_ids in method_specs:
        for operator in operators:
            try:
                kernels = load_submission_artifacts(
                    method,
                    operator_id=operator.op_id,
                    configuration_ids=list(configuration_ids) or None,
                    language=language,
                    cuda_version=str(backend.labels.get("cuda_version", "")),
                )
            except ValueError as exc:
                if "language" in str(exc) or "supports" in str(exc):
                    continue
                raise
            job_id = str(uuid5(NAMESPACE_URL, "|".join([
                args.campaign_id,
                args.backend,
                args.dataset_id,
                str(method.resolve()),
                operator.op_id,
                str(args.shard_index),
            ])))
            expected = fingerprint(benchmark, backend, kernels, operator, all_shard_matrices)
            job = runtime.db.get_job(job_id)
            if job is None:
                job = JobRecord(
                    job_id=job_id,
                    generator_id="spmv-campaign",
                    backends=[args.backend],
                    suites=["spmv"],
                    dataset_id=args.dataset_id,
                    kernels=kernels,
                    operator_ids=[operator.op_id],
                    tags={"campaign_id": args.campaign_id, "fingerprint": expected},
                    started_at=utc_now_iso(),
                )
                runtime.db.upsert_job(job)
            elif job.tags.get("fingerprint") != expected:
                if not args.adopt_fingerprint:
                    raise ValueError("source, protocol, dataset, or worker changed; use a new campaign ID")
                job.tags["fingerprint"] = expected
                job.kernels = kernels
                runtime.db.upsert_job(job)
            jobs.append((job, operator))
    if not jobs:
        raise ValueError(f"no compatible SpMV submission/operator combinations for {language}")
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else float("inf")
    stop = []

    def request_stop(signum, frame):
        stop.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    for job, operator in jobs:
        rows = runtime.db.query_results(job_ids=[job.job_id])
        completed = completed_keys(rows, args.retry_failures)
        for kernel in job.kernels:
            for matrix in matrices:
                key = (*kernel_key(kernel), matrix.matrix_id)
                if key in completed:
                    continue
                if stop or time.monotonic() >= deadline:
                    persist_checkpoint(runtime, job)
                    print(json.dumps({"status": "checkpointed", "job_id": job.job_id}), flush=True)
                    return 75
                try:
                    result = benchmark.run_case(backend, job, operator, matrix, kernel)
                except Exception as exc:
                    result = failed_result(backend, benchmark, job, operator, matrix, kernel, exc)
                results = result if isinstance(result, list) else [result]
                for item in results:
                    runtime.db.insert_result(item)
                    completed.add(row_key(item.model_dump()))
                print(json.dumps({
                    "dtype": operator.dtype,
                    "method": kernel.name,
                    "matrix": matrix.matrix_id,
                    "status": [item.status.value for item in results],
                }), flush=True)
            persist_checkpoint(runtime, job)
        target = {
            (*kernel_key(kernel), matrix.matrix_id)
            for kernel in job.kernels
            for matrix in matrices
        }
        covered = {row_key(row) for row in runtime.db.query_results(job_ids=[job.job_id])}
        persist_checkpoint(runtime, job, complete=target.issubset(covered))
    print(json.dumps({"status": "complete", "jobs": [job.job_id for job, _ in jobs]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Checkpointed CSR SpMV campaign")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset-id", default="suitesparse_all")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--submission", action="append", type=Path, default=[])
    parser.add_argument("--configuration-id", action="append", default=[])
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--adopt-fingerprint", action="store_true",
                        help="adopt the current source and dataset fingerprint for this named campaign")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be within shard-count")
    runtime = create_runtime(args.config)
    lock = runtime.db.path.with_suffix(".campaign.lock")
    with lock.open("a+b") as handle:
        if os.name != "nt":
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_campaign(runtime, args)


if __name__ == "__main__":
    raise SystemExit(main())

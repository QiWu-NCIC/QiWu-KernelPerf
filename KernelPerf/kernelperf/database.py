from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import BenchmarkResult, CaseStatus, JobRecord, JobStatus, WorkerNode


class PerfDatabase:
    def __init__(self, path: str | Path = "data/kernelperf.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 30000")
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("pragma journal_mode = wal")
            conn.executescript(
                """
                create table if not exists jobs (
                  job_id text primary key,
                  generator_id text not null,
                  submitter text,
                  status text not null,
                  priority integer not null,
                  created_at text not null,
                  started_at text,
                  finished_at text,
                  assigned_worker text,
                  request_json text not null,
                  error text
                );

                create table if not exists workers (
                  worker_id text primary key,
                  backend_id text not null,
                  endpoint text not null,
                  labels_json text not null,
                  status text not null,
                  last_heartbeat text not null,
                  current_job_id text,
                  log_tail_json text not null
                );

                create table if not exists results (
                  result_id text primary key,
                  job_id text not null,
                  generator_id text not null,
                  backend_id text not null,
                  backend_kind text not null,
                  suite text not null,
                  operator_id text not null,
                  operator_name text not null,
                  matrix_id text not null,
                  matrix_name text not null,
                  rows integer not null,
                  cols integer not null,
                  nnz integer not null,
                  kernel_name text not null,
                  status text not null default 'pass',
                  preprocess_ms real not null default 0.0,
                  runtime_ms real not null,
                  gflops real not null,
                  arithmetic_intensity real not null,
                  timestamp text not null,
                  metadata_json text not null
                );

                create table if not exists job_logs (
                  log_id integer primary key autoincrement,
                  job_id text not null,
                  timestamp text not null,
                  worker_id text,
                  message text not null
                );

                create index if not exists idx_results_filter
                  on results(job_id, backend_id, suite, operator_id);
                create index if not exists idx_jobs_status
                  on jobs(status, priority, created_at);
                create index if not exists idx_job_logs_job
                  on job_logs(job_id, log_id);
                """
            )
            result_columns = {
                row["name"] for row in conn.execute("pragma table_info(results)")
            }
            if "status" not in result_columns:
                conn.execute(
                    "alter table results add column status text not null default 'pass'"
                )
            if "preprocess_ms" not in result_columns:
                conn.execute(
                    "alter table results add column preprocess_ms real not null default 0.0"
                )

    def upsert_job(self, job: JobRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into jobs(job_id, generator_id, submitter, status, priority, created_at,
                                 started_at, finished_at, assigned_worker, request_json, error)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(job_id) do update set
                  status=excluded.status,
                  started_at=excluded.started_at,
                  finished_at=excluded.finished_at,
                  assigned_worker=excluded.assigned_worker,
                  request_json=excluded.request_json,
                  error=excluded.error
                """,
                (
                    job.job_id,
                    job.generator_id,
                    job.submitter,
                    job.status.value,
                    job.priority,
                    job.created_at,
                    job.started_at,
                    job.finished_at,
                    job.assigned_worker,
                    job.model_dump_json(),
                    job.error,
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.connect() as conn:
            row = conn.execute("select request_json from jobs where job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord.model_validate_json(row["request_json"])

    def list_jobs(self, status: JobStatus | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "select * from jobs"
        args: list[Any] = []
        if status:
            query += " where status = ?"
            args.append(status.value)
        query += " order by created_at desc limit ?"
        args.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [dict(row) | {"request": json.loads(row["request_json"])} for row in rows]

    def append_job_log(
        self,
        job_id: str,
        timestamp: str,
        message: str,
        worker_id: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into job_logs(job_id, timestamp, worker_id, message)
                values (?, ?, ?, ?)
                """,
                (job_id, timestamp, worker_id, message),
            )

    def list_job_logs(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select log_id, timestamp, worker_id, message
                from job_logs
                where job_id = ?
                order by log_id
                """,
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_worker(self, worker: WorkerNode) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into workers(worker_id, backend_id, endpoint, labels_json, status,
                                    last_heartbeat, current_job_id, log_tail_json)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(worker_id) do update set
                  backend_id=excluded.backend_id,
                  endpoint=excluded.endpoint,
                  labels_json=excluded.labels_json,
                  status=excluded.status,
                  last_heartbeat=excluded.last_heartbeat,
                  current_job_id=excluded.current_job_id,
                  log_tail_json=excluded.log_tail_json
                """,
                (
                    worker.worker_id,
                    worker.backend_id,
                    worker.endpoint,
                    json.dumps(worker.labels),
                    worker.status.value,
                    worker.last_heartbeat,
                    worker.current_job_id,
                    json.dumps(worker.log_tail),
                ),
            )

    def delete_workers_not_in(self, worker_ids: set[str]) -> int:
        with self.connect() as conn:
            if not worker_ids:
                cursor = conn.execute("delete from workers")
            else:
                placeholders = ",".join("?" for _ in worker_ids)
                cursor = conn.execute(
                    f"delete from workers where worker_id not in ({placeholders})",
                    sorted(worker_ids),
                )
        return cursor.rowcount

    def list_workers(self, worker_ids: set[str] | None = None) -> list[dict[str, Any]]:
        query = "select * from workers"
        args: list[str] = []
        if worker_ids is not None:
            if not worker_ids:
                return []
            placeholders = ",".join("?" for _ in worker_ids)
            query += f" where worker_id in ({placeholders})"
            args.extend(sorted(worker_ids))
        query += " order by worker_id"
        with self.connect() as conn:
            rows = conn.execute(query, args).fetchall()
        workers = []
        for row in rows:
            item = dict(row)
            item["labels"] = json.loads(item.pop("labels_json"))
            item["log_tail"] = json.loads(item.pop("log_tail_json"))
            workers.append(item)
        return workers

    def insert_result(self, result: BenchmarkResult) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into results(result_id, job_id, generator_id, backend_id, backend_kind,
                                    suite, operator_id, operator_name, matrix_id, matrix_name,
                                    rows, cols, nnz, kernel_name, status, preprocess_ms,
                                    runtime_ms, gflops, arithmetic_intensity, timestamp,
                                    metadata_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id,
                    result.job_id,
                    result.generator_id,
                    result.backend_id,
                    result.backend_kind,
                    result.suite,
                    result.operator_id,
                    result.operator_name,
                    result.matrix_id,
                    result.matrix_name,
                    result.rows,
                    result.cols,
                    result.nnz,
                    result.kernel_name,
                    result.status.value,
                    result.preprocess_ms,
                    result.runtime_ms,
                    result.gflops,
                    result.arithmetic_intensity,
                    result.timestamp,
                    json.dumps(result.metadata),
                ),
            )

    def query_results(
        self,
        job_ids: list[str] | None = None,
        backend_ids: list[str] | None = None,
        suites: list[str] | None = None,
        operator_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        for column, values in [
            ("job_id", job_ids),
            ("backend_id", backend_ids),
            ("suite", suites),
            ("operator_id", operator_ids),
        ]:
            if values:
                clauses.append(f"{column} in ({','.join('?' for _ in values)})")
                args.extend(values)
        query = "select * from results"
        if clauses:
            query += " where " + " and ".join(clauses)
        query += " order by backend_id, suite, operator_id, matrix_name"
        with self.connect() as conn:
            rows = conn.execute(query, args).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["status"] = CaseStatus(item["status"]).value
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import BenchmarkResult, ContestCreateRequest, JobRecord


FAILED_CASE_GFLOPS = 1e-12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContestDatabase:
    """Separate materialized database for contest definitions and score snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 30000")
        conn.execute("pragma foreign_keys = on")
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("pragma journal_mode = wal")
            conn.executescript(
                """
                create table if not exists contests (
                  contest_id text primary key,
                  name text not null,
                  start_time text not null,
                  backend_id text not null,
                  dataset_id text not null,
                  suite text not null,
                  operator_id text not null,
                  operator_ids_json text,
                  created_at text not null
                );

                create table if not exists contest_entries (
                  contest_id text not null,
                  nickname text not null,
                  job_id text not null,
                  generator_id text not null,
                  submitted_at text not null,
                  status text not null,
                  finished_at text,
                  primary key(contest_id, nickname),
                  foreign key(contest_id) references contests(contest_id)
                );

                create table if not exists contest_results (
                  result_id text primary key,
                  contest_id text not null,
                  nickname text not null,
                  job_id text not null,
                  matrix_id text not null,
                  matrix_name text not null,
                  nnz integer not null,
                  status text not null,
                  gflops real not null,
                  preprocess_ms real not null default 0.0,
                  runtime_ms real not null,
                  timestamp text not null,
                  metadata_json text not null,
                  foreign key(contest_id, nickname)
                    references contest_entries(contest_id, nickname)
                    on delete cascade
                );

                create index if not exists idx_contest_results_entry
                  on contest_results(contest_id, nickname, job_id);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("pragma table_info(contests)").fetchall()
            }
            if "operator_ids_json" not in columns:
                conn.execute("alter table contests add column operator_ids_json text")
            result_columns = {
                row["name"]
                for row in conn.execute("pragma table_info(contest_results)").fetchall()
            }
            if "preprocess_ms" not in result_columns:
                conn.execute(
                    "alter table contest_results add column preprocess_ms real not null default 0.0"
                )

    def create_contest(self, contest: ContestCreateRequest) -> dict[str, Any]:
        start_time = contest.start_time.isoformat()
        operator_ids = contest.operator_ids or (
            [contest.operator_id] if contest.operator_id is not None else []
        )
        if not operator_ids:
            raise ValueError("all_operators must be resolved before persistence")
        operator_id = operator_ids[0] if len(operator_ids) == 1 else "*"
        created_at = (contest.created_at or datetime.now(timezone.utc)).isoformat()
        with self.connect() as conn:
            try:
                conn.execute(
                    """
                    insert into contests(contest_id, name, start_time, backend_id,
                                         dataset_id, suite, operator_id,
                                         operator_ids_json, created_at)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contest.contest_id,
                        contest.name,
                        start_time,
                        contest.backend_id,
                        contest.dataset_id,
                        contest.suite,
                        operator_id,
                        json.dumps(operator_ids),
                        created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"contest already exists: {contest.contest_id}") from exc
        created = self.get_contest(contest.contest_id)
        assert created is not None
        return created

    def get_contest(self, contest_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "select * from contests where contest_id = ?", (contest_id,)
            ).fetchone()
        return self._normalize_contest(row) if row is not None else None

    def list_contests(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from contests order by start_time desc, contest_id"
            ).fetchall()
        return [self._normalize_contest(row) for row in rows]

    @staticmethod
    def _normalize_contest(row: sqlite3.Row) -> dict[str, Any]:
        contest = dict(row)
        encoded = contest.pop("operator_ids_json", None)
        contest["operator_ids"] = (
            json.loads(encoded) if encoded else [contest["operator_id"]]
        )
        return contest

    def delete_contest(self, contest_id: str) -> bool:
        with self.connect() as conn:
            conn.execute(
                "delete from contest_results where contest_id = ?", (contest_id,)
            )
            conn.execute(
                "delete from contest_entries where contest_id = ?", (contest_id,)
            )
            cursor = conn.execute(
                "delete from contests where contest_id = ?", (contest_id,)
            )
        return cursor.rowcount > 0

    def register_submission(self, job: JobRecord, nickname: str) -> None:
        if job.contest_id is None:
            return
        with self.connect() as conn:
            conn.execute(
                "delete from contest_results where contest_id = ? and nickname = ?",
                (job.contest_id, nickname),
            )
            conn.execute(
                """
                insert into contest_entries(contest_id, nickname, job_id, generator_id,
                                            submitted_at, status, finished_at)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(contest_id, nickname) do update set
                  job_id=excluded.job_id,
                  generator_id=excluded.generator_id,
                  submitted_at=excluded.submitted_at,
                  status=excluded.status,
                  finished_at=excluded.finished_at
                """,
                (
                    job.contest_id,
                    nickname,
                    job.job_id,
                    job.generator_id,
                    job.created_at,
                    job.status.value,
                    job.finished_at,
                ),
            )

    def update_job(self, job: JobRecord) -> None:
        if job.contest_id is None:
            return
        nickname = job.nickname or job.generator_id
        with self.connect() as conn:
            conn.execute(
                """
                update contest_entries
                set status = ?, finished_at = ?
                where contest_id = ? and nickname = ? and job_id = ?
                """,
                (
                    job.status.value,
                    job.finished_at,
                    job.contest_id,
                    nickname,
                    job.job_id,
                ),
            )

    def insert_result(self, job: JobRecord, result: BenchmarkResult) -> None:
        if job.contest_id is None:
            return
        nickname = job.nickname or job.generator_id
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into contest_results(
                  result_id, contest_id, nickname, job_id, matrix_id, matrix_name,
                  nnz, status, gflops, preprocess_ms, runtime_ms, timestamp, metadata_json
                )
                select ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                where exists (
                  select 1 from contest_entries
                  where contest_id = ? and nickname = ? and job_id = ?
                )
                """,
                (
                    result.result_id,
                    job.contest_id,
                    nickname,
                    job.job_id,
                    result.matrix_id,
                    result.matrix_name,
                    result.nnz,
                    result.status.value,
                    result.gflops,
                    result.preprocess_ms,
                    result.runtime_ms,
                    result.timestamp,
                    json.dumps(result.metadata),
                    job.contest_id,
                    nickname,
                    job.job_id,
                ),
            )

    def leaderboard(self, contest_id: str) -> dict[str, Any] | None:
        contest = self.get_contest(contest_id)
        if contest is None:
            return None
        with self.connect() as conn:
            entries = conn.execute(
                """
                select * from contest_entries
                where contest_id = ?
                order by submitted_at, nickname
                """,
                (contest_id,),
            ).fetchall()
            result_rows = conn.execute(
                """
                select * from contest_results
                where contest_id = ?
                order by nickname, matrix_name
                """,
                (contest_id,),
            ).fetchall()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in result_rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            grouped.setdefault(item["nickname"], []).append(item)

        participants: list[dict[str, Any]] = []
        for entry_row in entries:
            entry = dict(entry_row)
            rows = grouped.get(entry["nickname"], [])
            terminal = entry["status"] in {"succeeded", "failed", "cancelled"}
            score = self._geometric_mean(rows) if terminal else None
            participants.append(
                {
                    **entry,
                    "score_gflops": score,
                    "results": rows,
                }
            )

        participants.sort(
            key=lambda item: (
                item["score_gflops"] is None,
                -(item["score_gflops"] or 0.0),
                item["submitted_at"],
                item["nickname"],
            )
        )
        rank = 0
        for participant in participants:
            if participant["score_gflops"] is None:
                participant["rank"] = None
            else:
                rank += 1
                participant["rank"] = rank
        return {"contest": contest, "participants": participants}

    @staticmethod
    def _geometric_mean(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return FAILED_CASE_GFLOPS
        values = [
            row["gflops"]
            if row["status"] == "pass" and row["gflops"] > 0
            else FAILED_CASE_GFLOPS
            for row in rows
        ]
        return math.exp(math.fsum(math.log(value) for value in values) / len(values))

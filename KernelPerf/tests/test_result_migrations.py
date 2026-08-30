from __future__ import annotations

import sqlite3

from kernelperf.contest import ContestDatabase
from kernelperf.database import PerfDatabase
from kernelperf.models import BenchmarkResult, CaseStatus


def test_perf_database_migrates_preprocess_ms_and_persists_new_values(tmp_path):
    path = tmp_path / "perf.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table results (
          result_id text primary key, job_id text not null, generator_id text not null,
          backend_id text not null, backend_kind text not null, suite text not null,
          operator_id text not null, operator_name text not null, matrix_id text not null,
          matrix_name text not null, rows integer not null, cols integer not null,
          nnz integer not null, kernel_name text not null, status text not null,
          runtime_ms real not null, gflops real not null,
          arithmetic_intensity real not null, timestamp text not null,
          metadata_json text not null
        );
        insert into results values (
          'old', 'job', 'generator', 'backend', 'gpu', 'spmv', 'spmv.csr.fp32',
          'SpMV', 'matrix', 'matrix', 1, 1, 1, 'candidate', 'pass', 1.0, 2.0,
          0.0, 'timestamp', '{}'
        );
        """
    )
    connection.commit()
    connection.close()

    database = PerfDatabase(path)
    assert database.query_results()[0]["preprocess_ms"] == 0.0

    database.insert_result(_result("new", preprocess_ms=4.25))
    rows = {row["result_id"]: row for row in database.query_results()}
    assert rows["new"]["preprocess_ms"] == 4.25


def test_contest_database_migrates_preprocess_ms_column(tmp_path):
    path = tmp_path / "contest.sqlite"
    database = ContestDatabase(path)
    connection = database.connect()
    connection.execute("alter table contest_results rename to contest_results_new")
    connection.executescript(
        """
        create table contest_results (
          result_id text primary key, contest_id text not null, nickname text not null,
          job_id text not null, matrix_id text not null, matrix_name text not null,
          nnz integer not null, status text not null, gflops real not null,
          runtime_ms real not null, timestamp text not null, metadata_json text not null,
          foreign key(contest_id, nickname)
            references contest_entries(contest_id, nickname) on delete cascade
        );
        drop table contest_results_new;
        """
    )
    connection.commit()
    connection.close()

    database = ContestDatabase(path)
    columns_connection = database.connect()
    columns = {
        row["name"]
        for row in columns_connection.execute("pragma table_info(contest_results)")
    }
    columns_connection.close()
    assert "preprocess_ms" in columns


def _result(result_id: str, preprocess_ms: float) -> BenchmarkResult:
    return BenchmarkResult(
        result_id=result_id,
        job_id="job",
        generator_id="generator",
        backend_id="backend",
        backend_kind="gpu",
        suite="spmv",
        operator_id="spmv.csr.fp32",
        operator_name="SpMV",
        matrix_id=result_id,
        matrix_name=result_id,
        rows=1,
        cols=1,
        nnz=1,
        kernel_name="candidate",
        status=CaseStatus.passed,
        preprocess_ms=preprocess_ms,
        runtime_ms=1.0,
        gflops=2.0,
        arithmetic_intensity=0.0,
    )

from __future__ import annotations

import math
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kernelperf.app import create_app
from kernelperf.contest import ContestDatabase, FAILED_CASE_GFLOPS
from kernelperf.models import (
    BenchmarkResult,
    CaseStatus,
    ContestCreateRequest,
    JobRecord,
    JobStatus,
    JobSubmitRequest,
    KernelArtifact,
)


KERNEL_SOURCE = Path("tests/fixtures/spmv_lifecycle.cu").read_text()

BENCHMARK_CONFIG = json.loads(Path("config/benchmarks.json").read_text())
DEFAULT_BENCHMARK = next(item for item in BENCHMARK_CONFIG if item.get("default"))
PRIVATE_BENCHMARK = next(item for item in BENCHMARK_CONFIG if not item.get("public"))
BACKEND_ID = json.loads(Path("config/workers.json").read_text())[0]["backend_id"]
DATASET_ID = DEFAULT_BENCHMARK["default_dataset_id"]
BENCHMARK_ID = DEFAULT_BENCHMARK["benchmark_id"]
OPERATOR_ID = DEFAULT_BENCHMARK["operators"][0]["op_id"]


def contest_payload(contest_id: str = "test-contest") -> dict[str, object]:
    return {
        "contest_id": contest_id,
        "name": "测试contest",
        "start_time": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "backend_id": BACKEND_ID,
        "dataset_id": DATASET_ID,
        "suite": BENCHMARK_ID,
        "operator_id": OPERATOR_ID,
    }


def test_contest_creation_is_admin_only_and_pages_are_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELPERF_ADMIN_TOKEN", "test-admin-token")
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    with TestClient(app) as client:
        denied = client.post("/api/v1/contests", json=contest_payload())
        created = client.post(
            "/api/v1/contests",
            json=contest_payload(),
            headers={"x-kernelperf-admin-token": "test-admin-token"},
        )
        contests = client.get("/api/v1/contests")
        contest_page = client.get("/contest")
        ranking_page = client.get("/contest/test-contest")
        dashboard = client.get("/")

    assert denied.status_code == 403
    assert created.status_code == 201
    assert created.json()["backend_id"] == BACKEND_ID
    assert [contest["contest_id"] for contest in contests.json()] == ["test-contest"]
    assert "contestRows" in contest_page.text
    assert "leaderboardRows" in ranking_page.text
    assert 'id="plots"' in dashboard.text
    assert contest_page.headers["cache-control"].startswith("no-store")


def test_contest_deletion_is_admin_only(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELPERF_ADMIN_TOKEN", "test-admin-token")
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    with TestClient(app) as client:
        client.post(
            "/api/v1/contests",
            json=contest_payload(),
            headers={"x-kernelperf-admin-token": "test-admin-token"},
        ).raise_for_status()
        denied = client.delete("/api/v1/contests/test-contest")
        deleted = client.delete(
            "/api/v1/contests/test-contest",
            headers={"x-kernelperf-admin-token": "test-admin-token"},
        )
        contests = client.get("/api/v1/contests")
        leaderboard = client.get("/api/v1/contests/test-contest/leaderboard")

    assert denied.status_code == 403
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "contest_id": "test-contest"}
    assert contests.json() == []
    assert leaderboard.status_code == 404


def test_contest_submission_infers_fixed_environment_from_database(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELPERF_ADMIN_TOKEN", "test-admin-token")
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    app.state.contest_db.create_contest(
        ContestCreateRequest.model_validate(contest_payload())
    )
    with pytest.raises(ValueError, match="requires the complete fixed dataset"):
        app.state.scheduler.submit(
            JobSubmitRequest.model_validate(
                {
                    "contest-id": "test-contest",
                    "nick-name": "alice",
                    "matrix_ids": ["case"],
                    "kernels": [{
                        "name": "candidate",
                        "source": KERNEL_SOURCE,
                        "metadata": {"operator_id": OPERATOR_ID, "base_format": "csr"},
                    }],
                }
            )
        )
    job, _ = app.state.scheduler.submit(
        JobSubmitRequest.model_validate(
            {
                "contest-id": "test-contest",
                "nick-name": "alice",
                "kernels": [{
                    "name": "candidate",
                    "source": KERNEL_SOURCE,
                    "metadata": {"operator_id": OPERATOR_ID, "base_format": "csr"},
                }],
            }
        )
    )

    assert job.generator_id == "alice"
    assert job.backends == [BACKEND_ID]
    assert job.dataset_id == DATASET_ID
    assert job.suites == [BENCHMARK_ID]
    assert job.operator_ids == [OPERATOR_ID]
    assert job.contest_id == "test-contest"
    assert job.nickname == "alice"


def test_contest_configuration_persists_in_separate_database(tmp_path):
    db_path = tmp_path / "contest.sqlite"
    ContestDatabase(db_path).create_contest(
        ContestCreateRequest.model_validate(contest_payload())
    )

    contest = ContestDatabase(db_path).get_contest("test-contest")

    assert contest is not None
    assert contest["backend_id"] == BACKEND_ID
    assert contest["dataset_id"] == DATASET_ID
    assert contest["suite"] == BENCHMARK_ID
    assert contest["operator_id"] == OPERATOR_ID
    assert contest["operator_ids"] == [OPERATOR_ID]


def test_suite_wide_contest_freezes_enabled_operators_and_created_at(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELPERF_ADMIN_TOKEN", "test-admin-token")
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    created_at = "2026-08-08T12:00:00+08:00"
    payload = contest_payload("private-all")
    payload.update(
        {
            "created_at": created_at,
            "backend_id": BACKEND_ID,
            "dataset_id": "none",
            "suite": PRIVATE_BENCHMARK["benchmark_id"],
            "all_operators": True,
        }
    )
    payload.pop("operator_id")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/contests",
            json=payload,
            headers={"x-kernelperf-admin-token": "test-admin-token"},
        )

    assert response.status_code == 201
    contest = response.json()
    assert contest["operator_id"] == "*"
    assert contest["created_at"] == created_at
    assert len(contest["operator_ids"]) == 10
    assert "cutlass.gemm" in contest["operator_ids"]
    assert "cute_dsl.jamba_attn_proj" in contest["operator_ids"]


def test_old_contest_schema_migrates_without_changing_existing_definition(tmp_path):
    db_path = tmp_path / "contest.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table contests (
              contest_id text primary key,
              name text not null,
              start_time text not null,
              backend_id text not null,
              dataset_id text not null,
              suite text not null,
              operator_id text not null,
              created_at text not null
            )
            """
        )
        conn.execute(
            """
            insert into contests values (
              'legacy', 'Legacy', '2026-07-22T12:00:00+08:00',
              'legacy-backend', 'legacy-dataset', 'legacy-benchmark',
              'legacy-operator', '2026-07-22T04:00:00+00:00'
            )
            """
        )

    contest = ContestDatabase(db_path).get_contest("legacy")

    assert contest is not None
    assert contest["operator_id"] == "legacy-operator"
    assert contest["operator_ids"] == ["legacy-operator"]


def test_multi_operator_contest_requires_exact_tagged_kernel_set(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELPERF_ADMIN_TOKEN", "test-admin-token")
    app = create_app(str(tmp_path / "perf.sqlite"), str(tmp_path / "contest.sqlite"))
    payload = contest_payload("multi-sol")
    payload.update(
        {
            "backend_id": BACKEND_ID,
            "dataset_id": "none",
            "suite": PRIVATE_BENCHMARK["benchmark_id"],
            "operator_ids": ["cutlass.gemm", "cute_dsl.jamba_attn_proj"],
        }
    )
    payload.pop("operator_id")
    kernel = {
        "name": "gemm",
        "source": "candidate",
        "metadata": {"problem_id": "cutlass.gemm"},
    }

    with TestClient(app) as client:
        client.post(
            "/api/v1/contests",
            json=payload,
            headers={"x-kernelperf-admin-token": "test-admin-token"},
        ).raise_for_status()
        incomplete = client.post(
            "/api/v1/jobs",
            json={
                "contest-id": "multi-sol",
                "nick-name": "alice",
                "kernels": [kernel],
            },
            headers={"x-kernelperf-admin-token": "test-admin-token"},
        )

    assert incomplete.status_code == 400
    assert "complete operator set" in incomplete.json()["detail"]


def test_failed_cases_are_near_zero_and_duplicate_nickname_replaces_snapshot(tmp_path):
    db = ContestDatabase(tmp_path / "contest.sqlite")
    db.create_contest(ContestCreateRequest.model_validate(contest_payload()))
    first = JobRecord(
        generator_id="generator",
        backends=[BACKEND_ID],
        suites=[BENCHMARK_ID],
        dataset_id=DATASET_ID,
        operator_ids=[OPERATOR_ID],
        kernels=[KernelArtifact(name="candidate", source=KERNEL_SOURCE)],
        contest_id="test-contest",
        nickname="alice",
        status=JobStatus.succeeded,
    )
    first.finished_at = datetime.now(timezone.utc).isoformat()
    db.register_submission(first, "alice")
    for index, (status, gflops) in enumerate(
        [(CaseStatus.passed, 8.0), (CaseStatus.error, 0.0), (CaseStatus.failed, 0.0)]
    ):
        db.insert_result(
            first,
            BenchmarkResult(
                job_id=first.job_id,
                generator_id=first.generator_id,
                backend_id=BACKEND_ID,
                backend_kind="mock",
                suite=BENCHMARK_ID,
                operator_id=OPERATOR_ID,
                operator_name="SpMV",
                matrix_id=f"matrix-{index}",
                matrix_name=f"matrix-{index}",
                rows=1,
                cols=1,
                nnz=index + 1,
                kernel_name="candidate",
                status=status,
                preprocess_ms=7.5,
                runtime_ms=1.0 if status == CaseStatus.passed else 0.0,
                gflops=gflops,
                arithmetic_intensity=0.0,
            ),
        )
    score = db.leaderboard("test-contest")["participants"][0]["score_gflops"]
    assert score == pytest.approx(math.pow(8.0 * FAILED_CASE_GFLOPS**2, 1 / 3))
    assert db.leaderboard("test-contest")["participants"][0]["results"][0]["preprocess_ms"] == 7.5

    replacement = first.model_copy(
        update={
            "job_id": "replacement-job",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    db.register_submission(replacement, "alice")
    participant = db.leaderboard("test-contest")["participants"][0]
    assert participant["job_id"] == "replacement-job"
    assert participant["results"] == []
    assert participant["score_gflops"] == FAILED_CASE_GFLOPS

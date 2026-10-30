import csv
import hashlib
import json
import sqlite3

from scripts.backfill_failure_reasons import backfill, recover_reason, result_key


def test_backfill_only_matches_same_attempt_and_preserves_existing_values(tmp_path):
    data = tmp_path / "public" / "data"
    data.mkdir(parents=True)
    rows = [dict(job_id="job", backend_id="gpu", operator_id="spmv.csr.fp32",
                 matrix_id="group/matrix", status="fail", timestamp="2026-10-20T12:00:00+00:00",
                 method_name="Method", configuration_id="alg1", candidate_group="library",
                 solve_ms="0.123456789"),
            dict(job_id="job", backend_id="gpu", operator_id="spmv.csr.fp32",
                 matrix_id="group/matrix", status="fail", timestamp="2026-10-21T12:00:00+00:00",
                 method_name="Method", configuration_id="alg1", candidate_group="library",
                 solve_ms="0")]
    path = data / "results.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {"submissions": [{"path": "data/results.csv", "csv_sha256": "old"}]}
    (data / "index.json").write_text(json.dumps(index))
    database = tmp_path / "logs.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE results (job_id, backend_id, operator_id, matrix_id, kernel_name, status, timestamp, metadata_json)")
        connection.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           ("job", "gpu", "spmv.csr.fp32", "group/matrix", "Method",
                            "fail", rows[0]["timestamp"], json.dumps({
                                "implementation": {"configuration_id": "alg1", "candidate_group": "library"},
                                "failure_stage": "setup", "error": "out of memory"})))
    report = backfill(data, [database], write=True)
    assert report["reason_sources"] == {"exact_log": 1, "unavailable": 1}
    with path.open(encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    assert actual[0]["failure_reason"] == "setup: out of memory"
    assert actual[1]["failure_reason"].startswith("historical reason unavailable")
    assert [{key: value for key, value in row.items() if key != "failure_reason"}
            for row in actual] == rows
    assert json.loads((data / "index.json").read_text())["submissions"][0]["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert backfill(data, [database], write=True)["totals"]["changed_files"] == 0


def test_backfill_rejects_different_configuration_and_rhs():
    row = {"status": "fail", "method_name": "Same name", "configuration_id": "alg1",
           "candidate_group": "library", "rhs_columns": "2"}
    wrong = {"method_name": "Same name", "configuration_id": "alg2",
             "candidate_group": "library", "reason": "wrong reason"}
    assert recover_reason(row, {result_key(row): [wrong]}, {})[1] == "unavailable"
    assert result_key(row) != result_key({**row, "rhs_columns": "4"})


def test_best_reason_requires_unique_exact_candidate_diagnostic():
    row = {"status": "fail", "selection_role": "best", "selected_from": "alg1,alg2",
           "candidate_group": "library"}
    record = {"method_name": "Method", "configuration_id": "alg1",
              "candidate_group": "library", "reason": "setup: out of memory"}
    assert recover_reason(row, {}, {result_key(row): [record]}) == (record["reason"], "candidate_csv")
    assert recover_reason(row, {}, {result_key(row): [record, {**record, "reason": "timeout"}]})[1] == "ambiguous"

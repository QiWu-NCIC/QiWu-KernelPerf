import assert from "node:assert/strict";
import test from "node:test";
import {
  choosePerMatrixBest,
  configurations,
  latestAttemptsPerMatrix,
} from "../scripts/rebuild-cusparse-spmv.mjs";

test("full cuSPARSE campaign lists exactly the requested 27 configurations", () => {
  assert.equal(configurations.length, 27);
  assert.deepEqual(configurations.slice(0, 9), [
    "coo-alg1", "coo-alg2", "coo-default",
    "csc-alg1", "csc-alg2", "csc-default",
    "csr-alg1", "csr-alg2", "csr-default",
  ]);
  assert.deepEqual(configurations.slice(-2), ["sell-nrows-alg1", "sell-nrows-default"]);
});

test("candidate attempts keep the latest pass without timing-based retry bias", () => {
  const selected = latestAttemptsPerMatrix([
    { matrix_id: "m", status: "pass", solve_ms: 1, timestamp: "2026-01-01T00:00:01Z" },
    { matrix_id: "m", status: "pass", solve_ms: 3, timestamp: "2026-01-01T00:00:02Z" },
    { matrix_id: "n", status: "fail", timestamp: "2026-01-01T00:00:03Z" },
    { matrix_id: "n", status: "error", timestamp: "2026-01-01T00:00:04Z" },
  ]);

  assert.deepEqual(selected.map((row) => [row.matrix_id, row.status, row.solve_ms]), [
    ["m", "pass", 3],
    ["n", "error", undefined],
  ]);
});

test("per-matrix BEST compares candidate timings and retains all-failed matrices", () => {
  const selected = choosePerMatrixBest([
    { matrix_id: "m", configuration_id: "coo-alg1", status: "pass", solve_ms: 2 },
    { matrix_id: "m", configuration_id: "sell-c32-default", status: "pass", solve_ms: 1 },
    { matrix_id: "n", configuration_id: "coo-alg1", status: "error", solve_ms: 0 },
    { matrix_id: "n", configuration_id: "csr-default", status: "fail", solve_ms: 0 },
  ]);

  assert.deepEqual(selected.map((row) => [row.matrix_id, row.configuration_id, row.status]), [
    ["m", "sell-c32-default", "pass"],
    ["n", "csr-default", "fail"],
  ]);
});

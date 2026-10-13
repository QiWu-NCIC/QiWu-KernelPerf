import test from "node:test";
import assert from "node:assert/strict";
import { assertBestWinner } from "../scripts/spmm-data.mjs";

test("BEST must select the fastest passing candidate for its timing metric", () => {
  const candidates = [
    { configuration_id: "csr-alg1", status: "pass", solve_ms: "2.5" },
    { configuration_id: "csr-alg2", status: "pass", solve_ms: "1.25" },
  ];

  assert.doesNotThrow(() => assertBestWinner({
    status: "pass", selected_configuration_id: "csr-alg2", solve_ms: "1.25",
  }, candidates, "solve_ms"));
  assert.throws(() => assertBestWinner({
    status: "pass", selected_configuration_id: "csr-alg1", solve_ms: "2.5",
  }, candidates, "solve_ms"), /fastest passing candidate/);
});

test("failed BEST is valid only when every candidate failed", () => {
  const failed = [{ configuration_id: "csr-alg1", status: "error", solve_ms: "0" }];

  assert.doesNotThrow(() => assertBestWinner({ status: "error" }, failed, "solve_ms"));
  assert.throws(() => assertBestWinner({ status: "fail" }, [
    ...failed, { configuration_id: "csr-alg2", status: "pass", solve_ms: "1" },
  ], "solve_ms"), /despite a passing candidate/);
  assert.throws(() => assertBestWinner({ status: "unknown" }, failed, "solve_ms"), /failed BEST/);
});

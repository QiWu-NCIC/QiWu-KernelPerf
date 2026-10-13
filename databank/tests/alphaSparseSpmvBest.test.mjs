import test from "node:test";
import assert from "node:assert/strict";
import { chooseBest } from "../scripts/rebuild-alphasparse-spmv.mjs";

test("AlphaSparse BEST chooses the fastest passing algorithm per matrix", () => {
  const best = chooseBest([
    { matrix_id: "a", configuration_id: "vector", status: "pass", solve_ms: "2" },
    { matrix_id: "a", configuration_id: "scalar", status: "pass", solve_ms: "1" },
    { matrix_id: "b", configuration_id: "vector", status: "error", solve_ms: "0" },
    { matrix_id: "b", configuration_id: "scalar", status: "pass", solve_ms: "0.5" },
  ]);

  assert.deepEqual(best.map((row) => [row.matrix_id, row.configuration_id]), [
    ["a", "scalar"], ["b", "scalar"],
  ]);
});

test("AlphaSparse BEST preserves all-failed matrices", () => {
  const best = chooseBest([
    { matrix_id: "a", configuration_id: "vector", status: "error", timestamp: "2026-01-01" },
    { matrix_id: "a", configuration_id: "scalar", status: "fail", timestamp: "2026-01-02" },
  ]);

  assert.equal(best.length, 1);
  assert.equal(best[0].status, "fail");
  assert.equal(best[0].configuration_id, "scalar");
});

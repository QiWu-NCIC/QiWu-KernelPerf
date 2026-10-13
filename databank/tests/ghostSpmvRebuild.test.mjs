import test from "node:test";
import assert from "node:assert/strict";
import { chooseOverallSigma, isCompleteCoverage, sigmaValues } from "../scripts/rebuild-ghost-sell-spmv.mjs";

test("GHOST sweep includes every configured sigma value", () => {
  assert.equal(sigmaValues.length, 18);
  assert.deepEqual(sigmaValues, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]);
});

test("overall sigma is selected by the cross-dtype geometric mean", () => {
  const candidates = new Map([
    ["1|fp32", [{ status: "pass", solve_only_efficiency_percent: "2" }]],
    ["1|fp64", [{ status: "pass", solve_only_efficiency_percent: "8" }]],
    ["2|fp32", [{ status: "pass", solve_only_efficiency_percent: "8" }]],
    ["2|fp64", [{ status: "pass", solve_only_efficiency_percent: "8" }]],
    ["4|fp32", [{ status: "fail", solve_only_efficiency_percent: "100" }]],
  ]);
  const selected = chooseOverallSigma(candidates);
  assert.equal(selected.sigma, 2);
  assert.ok(Math.abs(selected.score - 8) < 1e-12);
});

test("coverage requires every expected matrix exactly once", () => {
  const expected = new Set(["m1", "m2"]);
  assert.equal(isCompleteCoverage([{ matrix_id: "m1" }, { matrix_id: "m2" }], expected), true);
  assert.equal(isCompleteCoverage([{ matrix_id: "m1" }], expected), false);
  assert.equal(isCompleteCoverage([{ matrix_id: "m1" }, { matrix_id: "m1" }], expected), false);
});

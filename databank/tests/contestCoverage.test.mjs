import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {
  comparisonExclusions,
  comparisonMatrixCount,
  expectedMatrixCount,
  isComparisonExcluded,
} from "../src/utils/contestCoverage.js";

const datasets = JSON.parse(fs.readFileSync(new URL("../../KernelPerf/config/datasets.json", import.meta.url), "utf8"));

test("partial full-corpus runs retain the complete denominator for both dtypes", () => {
  assert.equal(expectedMatrixCount("suitesparse_all", 185, datasets), 2904);
  assert.equal(expectedMatrixCount("suitesparse_all", 0, datasets), 2904);
  assert.equal(expectedMatrixCount("suitesparse_all", 2903, datasets), 2904);
});

test("sample datasets keep their declared coverage", () => {
  assert.equal(expectedMatrixCount("suitesparse_sample_100", 99, datasets), 100);
  assert.equal(expectedMatrixCount("suitesparse_validation_100", 100, datasets), 100);
});

test("undeclared datasets use observed coverage", () => {
  assert.equal(expectedMatrixCount("custom", 42, datasets), 42);
  assert.equal(expectedMatrixCount("none", 3, datasets), 3);
});

test("the reviewed exclusion list contains six actual full-corpus FP32 cases", () => {
  const policy = comparisonExclusions("suitesparse_all", "spmv", "fp32", datasets);
  const manifest = JSON.parse(fs.readFileSync(new URL("../../KernelPerf/config/datasets/suitesparse_all.json", import.meta.url), "utf8").replace(/^\uFEFF/, ""));
  const matrixIds = new Set(manifest.map((matrix) => matrix.matrix_id));
  assert.equal(policy.matrix_ids.length, 6);
  assert.equal(new Set(policy.matrix_ids).size, 6);
  assert.ok(policy.reason.includes("not a correctness pass"));
  for (const matrixId of policy.matrix_ids) assert.ok(matrixIds.has(matrixId), matrixId);
});

test("comparison exclusions apply to every method, platform and original status", () => {
  const matrixIds = comparisonExclusions("suitesparse_all", "spmv", "fp32", datasets).matrix_ids;
  for (const backend of ["A100-SXM4-80GB", "H100-SXM5-80GB", "RTX5090-SL3061", "BW1000", "Z100"]) {
    for (const method of ["alphasparse-adaptive", "cusparse-csr-alg1", "cusparse-best", "ghost-sell-best"]) {
      for (const status of ["pass", "error", "fail"]) {
        for (const matrixId of matrixIds) {
          const row = { dataset_id: "suitesparse_all", operator_id: "spmv.csr.fp32", dtype: "fp32", matrix_id: matrixId, status, backend_id: backend, method_id: method };
          const original = { ...row };
          assert.equal(isComparisonExcluded(row, datasets), true);
          assert.deepEqual(row, original);
        }
      }
    }
  }
});

test("unreviewed precision errors and other comparison scopes remain visible", () => {
  const row = { dataset_id: "suitesparse_all", operator_id: "spmv.csr.fp32", dtype: "fp32", matrix_id: "Sandia/ASIC_320ks", status: "error" };
  for (const changes of [
    { matrix_id: "Rajat/rajat24" },
    { matrix_id: "DIMACS10/delaunay_n21" },
    { matrix_id: "DIMACS10/rgg_n_2_21_s0" },
    { operator_id: "spmv.csr.fp64", dtype: "fp64" },
    { operator_id: "spmm.csr.fp32" },
    { dataset_id: "suitesparse_sample_100" },
    { dataset_id: "suitesparse_validation_100" },
  ]) assert.equal(isComparisonExcluded({ ...row, ...changes }, datasets), false);
});

test("excluded cases are removed from coverage without shrinking partial runs", () => {
  for (const observed of [0, 185, 2898]) {
    assert.equal(comparisonMatrixCount("suitesparse_all", "spmv", "fp32", observed, datasets), 2898);
    assert.equal(comparisonMatrixCount("suitesparse_all", "spmv", "fp64", observed, datasets), 2904);
  }
  assert.equal(comparisonMatrixCount("suitesparse_all", "spmm", "fp32", 185, datasets), 2904);
  assert.equal(comparisonMatrixCount("suitesparse_sample_100", "spmv", "fp32", 99, datasets), 100);
  assert.equal(comparisonMatrixCount("custom", "spmv", "fp32", 42, datasets), 42);
});

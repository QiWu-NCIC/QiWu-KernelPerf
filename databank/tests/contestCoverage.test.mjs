import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { expectedMatrixCount } from "../src/utils/contestCoverage.js";

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

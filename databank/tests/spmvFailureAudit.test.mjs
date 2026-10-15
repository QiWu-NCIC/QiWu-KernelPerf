import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const sampleMatrices = JSON.parse(fs.readFileSync(
  path.join(repositoryRoot, "KernelPerf/config/datasets/suitesparse_sample_100.json"), "utf8",
));

test("SpMV failure audit reports fully covered high-failure configurations", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "spmv-audit-"));
  try {
    const directory = path.join(root, "data/candidate-pool/spmv/test/dataset");
    fs.mkdirSync(directory, { recursive: true });
    const file = path.join(directory, "sample.csv");
    const rows = ["matrix_id,status"];
    rows.push(...sampleMatrices.map((matrix, index) => `${matrix.matrix_id},${index ? "fail" : "pass"}`));
    fs.writeFileSync(file, `${rows.join("\n")}\n`);
    fs.writeFileSync(path.join(root, "data/candidate-pool/index.json"), JSON.stringify({
      submissions: [{
        backend_id: "test",
        dataset_id: "suitesparse_sample_100",
        candidate_group: "GHOST-SELL-C32-sigma",
        configuration_id: "sigma-1",
        dtype: "fp32",
        path: "data/candidate-pool/spmv/test/dataset/sample.csv",
      }],
    }));
    const script = path.resolve("scripts/audit-spmv-failures.mjs");
    const result = spawnSync(process.execPath, [script, root, "0.2"], { encoding: "utf8" });
    assert.equal(result.status, 2, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.equal(report.severe.length, 1);
    assert.equal(report.severe[0].failures, 99);
    assert.equal(report.incomplete.length, 0);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("SpMV failure audit excludes isolated failures from incomplete-sweep alerts", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "spmv-audit-"));
  try {
    const directory = path.join(root, "data/candidate-pool/spmv/test/dataset");
    fs.mkdirSync(directory, { recursive: true });
    const rows = ["matrix_id,status", `${sampleMatrices[0].matrix_id},fail`];
    fs.writeFileSync(path.join(directory, "sample.csv"), `${rows.join("\n")}\n`);
    fs.writeFileSync(path.join(root, "data/candidate-pool/index.json"), JSON.stringify({
      submissions: [{
        backend_id: "test",
        dataset_id: "suitesparse_sample_100",
        candidate_group: "GHOST-SELL-C32-sigma",
        configuration_id: "sigma-1",
        dtype: "fp32",
        path: "data/candidate-pool/spmv/test/dataset/sample.csv",
      }],
    }));
    const script = path.resolve("scripts/audit-spmv-failures.mjs");
    const result = spawnSync(process.execPath, [script, root, "0.2"], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.equal(report.severe.length, 0);
    assert.equal(report.incomplete.length, 1);
    assert.equal(report.incomplete[0].missing_matrices, 99);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

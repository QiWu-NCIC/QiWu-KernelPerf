import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseCsv } from "./rebuild-cusparse-spmv.mjs";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const publicRoot = path.resolve(process.argv[2] || "public");
const threshold = Number(process.argv[3] || 0.2);
if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
  throw new Error("failure threshold must be between 0 and 1");
}

const datasetSpecs = JSON.parse(fs.readFileSync(
  path.join(repositoryRoot, "KernelPerf/config/datasets.json"), "utf8",
));
const expectedMatrices = new Map();
for (const dataset of datasetSpecs) {
  if (!dataset.manifest) continue;
  const manifestPath = path.join(repositoryRoot, "KernelPerf", dataset.manifest);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8").replace(/^\uFEFF/, ""));
  expectedMatrices.set(dataset.dataset_id, new Set(manifest.map((item) => item.matrix_id)));
}
const indexPath = path.join(publicRoot, "data/candidate-pool/index.json");
const index = JSON.parse(fs.readFileSync(indexPath, "utf8"));
const groups = new Map();

for (const entry of index.submissions || []) {
  const filename = path.join(publicRoot, entry.path.replaceAll("/", path.sep));
  if (!fs.existsSync(filename)) continue;
  const { rows } = parseCsv(fs.readFileSync(filename, "utf8"));
  const key = [
    entry.backend_id,
    entry.dataset_id,
    entry.candidate_group || entry.method_id,
    entry.configuration_id,
    entry.dtype,
  ].join("|");
  const group = groups.get(key) || {
    backend: entry.backend_id,
    dataset: entry.dataset_id,
    method: entry.candidate_group || entry.method_id,
    configuration: entry.configuration_id,
    dtype: entry.dtype,
    files: [],
    rows: [],
  };
  group.files.push(entry.path);
  group.rows.push(...rows);
  groups.set(key, group);
}

const incomplete = [];
const severe = [];
for (const group of groups.values()) {
  const expectedIds = expectedMatrices.get(group.dataset);
  const expected = expectedIds?.size;
  const matrixIds = new Set(group.rows.map((row) => row.matrix_id).filter(Boolean));
  const passes = group.rows.filter((row) => row.status === "pass").length;
  const failures = group.rows.length - passes;
  const missing = expectedIds ? [...expectedIds].filter((matrix) => !matrixIds.has(matrix)) : [];
  const unexpected = expectedIds ? [...matrixIds].filter((matrix) => !expectedIds.has(matrix)) : [];
  const report = {
    backend: group.backend,
    dataset: group.dataset,
    method: group.method,
    configuration: group.configuration,
    dtype: group.dtype,
    rows: group.rows.length,
    matrices: matrixIds.size,
    expected,
    passes,
    failures,
    failure_rate: group.rows.length ? failures / group.rows.length : 0,
    missing_matrices: missing.length,
    missing_matrix_sample: missing.slice(0, 20),
    unexpected_matrices: unexpected.length,
    files: group.files,
  };
  if (expected && report.failure_rate >= threshold
      && failures >= Math.max(10, Math.ceil(expected * 0.05))) {
    severe.push(report);
  }
  if (!expected || group.rows.length !== expected || matrixIds.size !== expected
      || missing.length || unexpected.length) {
    incomplete.push(report);
  }
}

console.log(JSON.stringify({ threshold, audited: groups.size, severe, incomplete }, null, 2));
if (severe.length) process.exitCode = 2;

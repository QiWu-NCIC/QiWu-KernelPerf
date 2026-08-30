import fs from "node:fs";
import path from "node:path";

const [resultsPath, casesPath, outputRoot] = process.argv.slice(2);
if (!resultsPath || !casesPath || !outputRoot) {
  console.error("Usage: node scripts/import-kernelperf-regression.mjs RESULTS_JSONL CASES_TSV OUTPUT_ROOT");
  process.exit(1);
}

const cases = new Map();
for (const line of fs.readFileSync(casesPath, "utf8").split(/\r?\n/)) {
  if (!line.trim()) continue;
  const [matrixId, rows, cols] = line.split("\t");
  cases.set(matrixId, { matrixId, rows: Number(rows), cols: Number(cols) });
}

const methods = {
  adaptive_fp32: { methodId: "csr-adaptive-cuda", methodName: "CSR-Adaptive CUDA", baseFormat: "csr", configurationId: "cuda-adaptive", dtype: "fp32", peak: 19500 },
  adaptive_fp64: { methodId: "csr-adaptive-cuda", methodName: "CSR-Adaptive CUDA", baseFormat: "csr", configurationId: "cuda-adaptive", dtype: "fp64", peak: 9700 },
  csr5_fp32: { methodId: "csr5", methodName: "CSR5", baseFormat: "csr", configurationId: "csr5", dtype: "fp32", peak: 19500 },
  csr5_fp64: { methodId: "csr5", methodName: "CSR5", baseFormat: "csr", configurationId: "csr5", dtype: "fp64", peak: 9700 },
  ghost_fp32: { methodId: "ghost-sell-c32-sigma-128", methodName: "GHOST SELL-C-sigma (C=32, sigma=128)", baseFormat: "sell", configurationId: "sigma-128", dtype: "fp32", peak: 19500 },
  ghost_fp64: { methodId: "ghost-sell-c32-sigma-128", methodName: "GHOST SELL-C-sigma (C=32, sigma=128)", baseFormat: "sell", configurationId: "sigma-128", dtype: "fp64", peak: 9700 },
};
const jobId = "adapter-regression-20260824";
const datasetId = "suitesparse_sample_100";
const backendId = "A100-SXM4-80GB";
const hardware = "A100-SXM4-80GB";
const timestamp = "2026-08-25T00:00:00.000Z";
const columns = [
  "schema_version", "submission_id", "method_id", "method_name", "configuration_id",
  "candidate_group", "selection_role", "selected_from", "base_format", "operator_id",
  "dtype", "backend_id", "hardware", "peak_gflops", "job_id", "dataset_id",
  "matrix_id", "matrix_name", "rows", "cols", "nnz", "status", "operations",
  "preprocess_ms", "solve_ms", "solve_gflops", "solve_only_efficiency_percent", "timestamp", "source_kind",
];

const records = fs.readFileSync(resultsPath, "utf8").split(/\r?\n/)
  .filter(Boolean).map((line) => JSON.parse(line));
const grouped = new Map();
for (const record of records) {
  const method = methods[record.binary];
  const matrix = cases.get(record.matrix_id);
  if (!method || !matrix) throw new Error(`unknown result mapping: ${record.binary} ${record.matrix_id}`);
  const key = record.binary;
  if (!grouped.has(key)) grouped.set(key, []);
  const metadata = record.metadata || {};
  const operations = Number(record.operations ?? metadata.operations ?? 0);
  const solveMs = Number(record.runtime_ms ?? 0);
  const passing = record.status === "ok" && record.valid === true;
  const status = record.status !== "ok" ? "error" : passing ? "pass" : "fail";
  const solveGflops = solveMs > 0 ? operations / solveMs / 1e6 : 0;
  grouped.get(key).push({
    schema_version: 2,
    submission_id: `${jobId}-${key}`,
    method_id: method.methodId,
    method_name: method.methodName,
    configuration_id: method.configurationId,
    candidate_group: "baseline-regression",
    selection_role: "candidate",
    selected_from: "",
    base_format: method.baseFormat,
    operator_id: `spmv.csr.${method.dtype}`,
    dtype: method.dtype,
    backend_id: backendId,
    hardware,
    peak_gflops: method.peak,
    job_id: jobId,
    dataset_id: datasetId,
    matrix_id: matrix.matrixId,
    matrix_name: matrix.matrixId.split("/").at(-1),
    rows: matrix.rows,
    cols: matrix.cols,
    nnz: Number(metadata.actual_nnz ?? operations / 2),
    status,
    operations,
    preprocess_ms: Number(record.preprocess_ms ?? 0),
    solve_ms: solveMs,
    solve_gflops: solveGflops,
    solve_only_efficiency_percent: passing ? solveGflops / method.peak * 100 : 0,
    timestamp,
    source_kind: "source",
  });
}

function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

fs.mkdirSync(outputRoot, { recursive: true });
for (const [binary, rows] of grouped) {
  const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => csvValue(row[column])).join(","))].join("\n") + "\n";
  const target = path.join(outputRoot, `${jobId}-${binary}.csv`);
  fs.writeFileSync(target, csv);
  console.log(JSON.stringify({ binary, rows: rows.length, target }));
}

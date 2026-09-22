import fs from "node:fs";
import path from "node:path";

export const schema = "kernelperf-spmm-v3";
export const timingColumns = {
  "solve-only": "solve_ms",
  "pre-plus-solve": "pre_plus_solve_ms",
  "pre-amortized": "pre_amortized_ms",
};
export const identityColumns = [
  "schema_version", "submission_id", "method_id", "method_name", "configuration_id",
  "candidate_group", "selection_role", "selection_metric", "selected_from", "base_format", "format",
  "operator_id", "dtype", "backend_id", "hardware", "peak_gflops", "dataset_id",
  "rhs_columns", "dense_layout", "op_a", "op_b", "alpha", "beta", "input_seed", "ranking_scope",
];
export const requiredColumns = [...identityColumns,
  "matrix_id", "matrix_name", "rows", "cols", "nnz", "status", "validation_status",
  "failed_elements", "invalid_elements", "nan_elements", "inf_elements", "operations",
  "preprocess_ms", ...Object.values(timingColumns), "solve_gflops", "pre_plus_solve_gflops",
  "pre_amortized_gflops", "solve_only_efficiency_percent", "pre_plus_solve_efficiency_percent",
  "pre_amortized_efficiency_percent", "library_version", "cpu_model", "selected_configuration_id",
  "error", "error_type", "failure_stage", "warmup", "iterations", "validation_safety_factor", "public_ranked",
];

export function safePath(value) {
  return typeof value === "string" && Boolean(value) && !value.includes("\\")
    && !value.includes(":") && value.split("/").every((part) => part && part !== "." && part !== "..");
}

export function parseCsv(text) {
  const records = [];
  let cells = [], cell = "", quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') { cell += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) { cells.push(cell); cell = ""; }
    else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      cells.push(cell);
      if (cells.some(Boolean)) records.push(cells);
      cells = []; cell = "";
    } else cell += character;
  }
  if (quoted) throw new Error("unterminated CSV quote");
  if (cells.length || cell) { cells.push(cell); records.push(cells); }
  if (records.length < 2) throw new Error("CSV needs a header and data rows");
  const headers = records.shift();
  headers[0] = headers[0].replace(/^\uFEFF/, "");
  if (new Set(headers).size !== headers.length) throw new Error("duplicate CSV columns");
  for (const name of requiredColumns) if (!headers.includes(name)) throw new Error("missing column: " + name);
  return records.map((values, index) => {
    if (values.length !== headers.length) throw new Error("CSV width mismatch at row " + (index + 2));
    return Object.fromEntries(headers.map((name, column) => [name, values[column]]));
  });
}

export function validateRows(rows, { official = true } = {}) {
  const first = rows[0];
  const matrices = new Set();
  const close = (actual, expected) => Math.abs(actual - expected) <= Math.max(1e-12, Math.abs(expected) * 1e-8);
  for (const row of rows) {
    for (const name of identityColumns) if (row[name] !== first[name]) throw new Error("mixed identity: " + name);
    if (row.schema_version !== "3" || row.operator_id !== "spmm.csr." + row.dtype) throw new Error("invalid SpMM schema/operator");
    if (!["fp32", "fp64"].includes(row.dtype)) throw new Error("unsupported dtype");
    if (![1, 2, 4, 8, 16, 32, 64, 128].includes(Number(row.rhs_columns))) throw new Error("invalid RHS count");
    if (row.ranking_scope !== (Number(row.rhs_columns) === 1 ? "sanity" : "main")) throw new Error("invalid ranking scope");
    if (row.dense_layout !== "row-major" || row.op_a !== "N" || row.op_b !== "N" || Number(row.alpha) !== 1 || Number(row.beta) !== 0) throw new Error("protocol mismatch");
    if (row.format !== "csr" || !["csr", "manual-selection"].includes(row.base_format)) throw new Error("invalid P0 format");
    if (official && (Number(row.warmup) !== 5 || Number(row.iterations) !== 20 || Number(row.input_seed) !== 20260922 || Number(row.validation_safety_factor) !== 4)) throw new Error("official measurement protocol mismatch");
    if (matrices.has(row.matrix_id)) throw new Error("duplicate matrix: " + row.matrix_id);
    matrices.add(row.matrix_id);
    for (const name of ["rows", "cols", "nnz", "rhs_columns", "peak_gflops"]) if (!Number.isFinite(Number(row[name])) || Number(row[name]) <= 0) throw new Error("invalid " + name);
    for (const name of ["preprocess_ms", ...Object.values(timingColumns), "failed_elements", "invalid_elements", "nan_elements", "inf_elements"]) if (!Number.isFinite(Number(row[name])) || Number(row[name]) < 0) throw new Error("invalid " + name);
    if (!close(Number(row.operations), 2 * Number(row.nnz) * Number(row.rhs_columns))) throw new Error("incorrect FLOP count");
    if (!["pass", "error", "fail"].includes(row.status)) throw new Error("invalid status");
    if (row.status === "pass") {
      if (row.validation_status !== "pass" || Number(row.failed_elements) || Number(row.invalid_elements)) throw new Error("inconsistent pass status");
      if (!row.library_version || row.library_version === "unknown") throw new Error("passing result has no library version");
      if (!row.method_name.includes(row.library_version)) throw new Error("method name is not versioned");
      if (!close(Number(row.pre_plus_solve_ms), Number(row.preprocess_ms) + Number(row.solve_ms))) throw new Error("pre+solve mismatch");
      if (!close(Number(row.pre_amortized_ms), Number(row.preprocess_ms) / Number(row.iterations) + Number(row.solve_ms))) throw new Error("amortization mismatch");
    }
    for (const [time, performance, efficiency] of [
      ["solve_ms", "solve_gflops", "solve_only_efficiency_percent"],
      ["pre_plus_solve_ms", "pre_plus_solve_gflops", "pre_plus_solve_efficiency_percent"],
      ["pre_amortized_ms", "pre_amortized_gflops", "pre_amortized_efficiency_percent"],
    ]) {
      const expected = row.status === "pass" ? Number(row.operations) / Number(row[time]) / 1e6 : 0;
      if (!Number.isFinite(expected) || !close(Number(row[performance]), expected) || !close(Number(row[efficiency]), expected / Number(row.peak_gflops) * 100)) throw new Error("invalid performance metrics");
    }
  }
  return rows;
}

export function readIndex(root) {
  const filename = path.join(root, "data/spmm/index.json");
  if (!fs.existsSync(filename)) return { schema_version: 3, result_schema: schema, submissions: [] };
  const index = JSON.parse(fs.readFileSync(filename, "utf8"));
  if (index.schema_version !== 3 || index.result_schema !== schema || !Array.isArray(index.submissions)) throw new Error("invalid SpMM catalog");
  return index;
}

export function scopeKey(row) {
  return [row.backend_id, row.dataset_id, row.operator_id, row.rhs_columns, row.dense_layout, row.candidate_group].join("|");
}

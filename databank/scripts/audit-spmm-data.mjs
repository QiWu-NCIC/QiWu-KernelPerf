import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { readIndex, parseCsv, validateRows, safePath, scopeKey, timingColumns } from "./spmm-data.mjs";

const root = path.resolve(process.argv[2] || "public");
const index = readIndex(root);
const candidates = new Map();
const bestRows = [];
const seen = new Set();
const sourceSeen = new Set();
const completeness = new Set();
for (const entry of index.submissions) {
  if (seen.has(entry.submission_id)) throw new Error("duplicate submission id");
  seen.add(entry.submission_id);
  if (!safePath(entry.path) || !entry.path.startsWith("data/spmm/results/")) throw new Error("unsafe SpMM result path");
  if (!safePath(entry.source_manifest) || !entry.source_manifest.startsWith("source/spmm/")) throw new Error("unsafe SpMM source path");
  const text = fs.readFileSync(path.join(root, entry.path), "utf8");
  if (crypto.createHash("sha256").update(text).digest("hex") !== entry.csv_sha256) throw new Error("CSV hash mismatch: " + entry.path);
  const rows = validateRows(parseCsv(text));
  for (const field of ["submission_id", "backend_id", "dataset_id", "dtype", "operator_id", "dense_layout", "configuration_id", "selection_role", "selection_metric"]) {
    if (rows[0][field] !== entry[field]) throw new Error("index identity mismatch: " + field);
  }
  if (Number(rows[0].rhs_columns) !== entry.rhs_columns) throw new Error("index RHS mismatch");
  if (entry.candidate_group === "cuSPARSE-SpMM-CSR" && entry.configuration_id === "csr-default" && entry.public_ranked) {
    throw new Error("cuSPARSE DEFAULT must not be publicly ranked");
  }
  const passed = rows.filter((row) => row.status === "pass").length;
  if (entry.passed !== passed || entry.failed !== rows.length - passed || entry.total !== rows.length) throw new Error("coverage count mismatch");
  if (!sourceSeen.has(entry.source_manifest)) {
    const manifestFile = path.join(root, entry.source_manifest);
    const plugin = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
    if (plugin.kind !== "qiwu-spmm-source-plugin" || plugin.source_sha256 !== entry.source_sha256) throw new Error("source identity mismatch");
    if (plugin.candidate_group !== entry.candidate_group || !plugin.supported_operators?.includes(entry.operator_id)) {
      throw new Error("source package does not match the result adapter or operator");
    }
    const digest = crypto.createHash("sha256");
    for (const file of [...plugin.files].sort()) {
      if (!safePath(file)) throw new Error("unsafe source path");
      digest.update(file).update("\0").update(fs.readFileSync(path.join(path.dirname(manifestFile), "files", file))).update("\0");
    }
    if (digest.digest("hex") !== plugin.source_sha256) throw new Error("source hash mismatch");
    sourceSeen.add(entry.source_manifest);
  }
  if (entry.selection_role === "candidate") {
    if (entry.dataset_id === "suitesparse_sample_100" && entry.rhs_columns !== 1 && rows.length === 100) {
      completeness.add([entry.backend_id, entry.dtype, entry.rhs_columns, entry.candidate_group, entry.configuration_id].join("|"));
    }
    for (const row of rows) {
      if (row.status !== "pass" || row.public_ranked !== "true" || row.ranking_scope !== "main") continue;
      const key = scopeKey(row) + "|" + row.matrix_id;
      if (!candidates.has(key)) candidates.set(key, []);
      candidates.get(key).push(row);
    }
  } else if (entry.selection_role === "best") bestRows.push(...rows);
  else throw new Error("unknown selection role");
}
for (const best of bestRows) {
  if (best.ranking_scope !== "main") throw new Error("sanity case entered BEST");
  const field = timingColumns[best.selection_metric];
  if (!field) throw new Error("BEST has unknown timing metric");
  const configurations = new Set(best.selected_from.split(","));
  const rows = (candidates.get(scopeKey(best) + "|" + best.matrix_id) || []).filter((row) => configurations.has(row.configuration_id));
  rows.sort((left, right) => Number(left[field]) - Number(right[field]) || left.configuration_id.localeCompare(right.configuration_id));
  const winner = rows[0];
  if (!winner || winner.configuration_id !== best.selected_configuration_id || Number(winner[field]) !== Number(best[field])) throw new Error("incorrect BEST winner for " + scopeKey(best) + "|" + best.matrix_id);
}
if (process.argv.includes("--require-complete")) {
  const platforms = [
    "A100-SXM4-80GB", "H100-SXM5-80GB", "RTX5090-SL3061",
    "BW1000-gfx936", "Z100-gfx906",
  ];
  for (const platform of platforms) {
    const isHip = ["BW1000-gfx936", "Z100-gfx906"].includes(platform);
    const groups = [
      ["AlphaSparse-SpMM-CSR", [1, 2, 3, 4, 5].map((value) => "csr-alg" + value)],
      isHip ? ["rocSPARSE-DTK26.04-SpMM-CSR", ["csr", "csr-row-split", "csr-nnz-split", "csr-merge-path"]]
        : ["cuSPARSE-SpMM-CSR", ["csr-default", "csr-alg1", "csr-alg2", "csr-alg3"]],
    ];
    for (const dtype of ["fp32", "fp64"]) for (const rhs of [2, 4, 8, 16, 32, 64, 128]) {
      for (const [group, configurations] of groups) for (const configuration of configurations) {
        const key = [platform, dtype, rhs, group, configuration].join("|");
        if (!completeness.has(key)) throw new Error("incomplete campaign: " + key);
      }
    }
  }
}
console.log(JSON.stringify({ operator: "spmm", submissions: seen.size, sources: sourceSeen.size, best_rows: bestRows.length, status: "pass" }));

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { parseCsv, presentationMethodName, readIndex, validateRows, timingColumns } from "./spmm-data.mjs";

const root = path.resolve(process.argv[2] || "public");
const index = readIndex(root);
const sourceByHash = new Map();
const sourceRoot = path.join(root, "source/spmm");
for (const directory of fs.readdirSync(sourceRoot, { withFileTypes: true })) {
  if (!directory.isDirectory()) continue;
  const manifestPath = path.join(sourceRoot, directory.name, "plugin.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  sourceByHash.set(manifest.source_sha256, {
    manifest,
    manifestPath: `source/spmm/${directory.name}/plugin.json`,
  });
}

const slug = (value) => String(value)
  .replace(/[^A-Za-z0-9._-]+/g, "-")
  .replace(/^-+|-+$/g, "")
  .slice(0, 160) || "result";

const readEntryRows = (entry) => {
  const text = fs.readFileSync(path.join(root, entry.path), "utf8");
  return { text, rows: validateRows(parseCsv(text)), entry };
};

const candidates = [];
for (const entry of index.submissions) {
  if (entry.selection_role !== "candidate" || entry.rhs_columns === 1) continue;
  const loaded = readEntryRows(entry);
  if (loaded.rows[0].ranking_scope !== "main") continue;
  candidates.push(loaded);
}

const scopeGroups = new Map();
for (const loaded of candidates) {
  const row = loaded.rows[0];
  const scope = [
    row.backend_id, row.dataset_id, row.operator_id, row.dtype,
    row.rhs_columns, row.dense_layout, row.candidate_group,
  ].join("|");
  if (!scopeGroups.has(scope)) scopeGroups.set(scope, []);
  scopeGroups.get(scope).push(loaded);
}

const oldBest = index.submissions.filter((entry) => entry.selection_role === "best");
const oldBestPaths = new Set(oldBest.map((entry) => entry.path));
const generated = [];
const generatedPaths = new Set();

for (const [scope, entries] of scopeGroups) {
  const first = entries[0].rows[0];
  const source = sourceByHash.get(entries[0].entry.source_sha256);
  if (!source) throw new Error(`no source package for ${first.candidate_group}/${entries[0].entry.source_sha256}`);

  const configurationIds = [...new Set(entries.map((item) => item.entry.configuration_id))].sort();
  const rowsByMatrix = new Map();
  for (const loaded of entries) {
    for (const row of loaded.rows) {
      if (row.status !== "pass" || row.public_ranked !== "true") continue;
      if (!rowsByMatrix.has(row.matrix_id)) rowsByMatrix.set(row.matrix_id, []);
      rowsByMatrix.get(row.matrix_id).push(row);
    }
  }
  const versions = [...new Set(entries.flatMap((item) => item.rows
    .filter((row) => row.status === "pass" && row.library_version && row.library_version !== "unknown")
    .map((row) => row.library_version)))].sort();
  const selectedFrom = configurationIds.join(",");

  for (const [selectionMetric, timingColumn] of Object.entries(timingColumns)) {
    const selected = [];
    for (const rows of rowsByMatrix.values()) {
      rows.sort((left, right) => Number(left[timingColumn]) - Number(right[timingColumn])
        || left.configuration_id.localeCompare(right.configuration_id));
      selected.push(rows[0]);
    }
    if (!selected.length) continue;
    selected.sort((left, right) => left.matrix_id.localeCompare(right.matrix_id));
    const methodId = slug(`${first.candidate_group}-best-${selectionMetric}`);
    const submissionId = slug([
      first.backend_id, first.dataset_id, first.dtype, `n${first.rhs_columns}`,
      first.dense_layout, first.candidate_group, selectionMetric,
    ].join("-"));
    const methodName = presentationMethodName({
      ...first,
      method_id: methodId,
      method_name: first.method_name,
      configuration_id: "per-matrix-best",
    });
    const rows = selected.map((row) => ({
      ...row,
      submission_id: submissionId,
      method_id: methodId,
      method_name: methodName,
      configuration_id: "per-matrix-best",
      candidate_group: first.candidate_group,
      selection_role: "best",
      selection_metric: selectionMetric,
      selected_from: selectedFrom,
      base_format: "manual-selection",
      source_kind: "derived",
      selected_configuration_id: row.configuration_id,
      public_ranked: "true",
    }));
    const headers = Object.keys(rows[0]);
    const escape = (value) => {
      const text = String(value ?? "");
      return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const text = [headers.join(","), ...rows.map((row) => headers.map((header) => escape(row[header])).join(","))].join("\n") + "\n";
    validateRows(parseCsv(text));
    const relative = [
      "data/results/spmm", first.backend_id, first.dataset_id, first.dtype,
      `n${first.rhs_columns}`, first.dense_layout, methodId, `${submissionId}.csv`,
    ].join("/");
    const target = path.join(root, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, text);
    generatedPaths.add(relative);
    const passed = rows.filter((row) => row.status === "pass").length;
    const metrics = {};
    for (const [metric, field] of Object.entries(timingColumns)) {
      metrics[metric] = passed ? rows.reduce((sum, row) => sum + Number(row[`${metric === "solve-only" ? "solve_only" : metric === "pre-plus-solve" ? "pre_plus_solve" : "pre_amortized"}_efficiency_percent`]), 0) / passed : 0;
    }
    generated.push({
      submission_id: submissionId,
      method_id: methodId,
      method_name: methodName,
      backend_id: first.backend_id,
      hardware: first.hardware,
      cpu_model: first.cpu_model,
      dataset_id: first.dataset_id,
      operator_id: first.operator_id,
      dtype: first.dtype,
      dense_layout: first.dense_layout,
      format: first.format,
      base_format: "manual-selection",
      configuration_id: "per-matrix-best",
      candidate_group: first.candidate_group,
      selection_role: "best",
      selection_metric: selectionMetric,
      ranking_scope: "main",
      rhs_columns: Number(first.rhs_columns),
      path: relative,
      source_manifest: source.manifestPath,
      source_sha256: source.manifest.source_sha256,
      csv_sha256: crypto.createHash("sha256").update(text).digest("hex"),
      public_ranked: true,
      passed,
      failed: rows.length - passed,
      total: rows.length,
      library_versions: versions,
      metrics,
    });
  }
}

for (const relative of oldBestPaths) {
  if (!generatedPaths.has(relative)) {
    const filename = path.join(root, relative);
    if (fs.existsSync(filename)) fs.unlinkSync(filename);
  }
}
index.submissions = [...index.submissions.filter((entry) => entry.selection_role !== "best"), ...generated]
  .sort((left, right) => left.path.localeCompare(right.path));
const indexPath = path.join(root, "data/results/spmm/index.json");
const temporary = `${indexPath}.tmp`;
fs.writeFileSync(temporary, JSON.stringify(index, null, 2) + "\n");
try {
  fs.renameSync(temporary, indexPath);
} catch (error) {
  if (process.platform !== "win32" || !["EPERM", "EEXIST"].includes(error.code)) throw error;
  fs.copyFileSync(temporary, indexPath);
  fs.unlinkSync(temporary);
}
console.log(JSON.stringify({ old_best: oldBest.length, generated_best: generated.length, candidates: candidates.length, status: "pass" }, null, 2));

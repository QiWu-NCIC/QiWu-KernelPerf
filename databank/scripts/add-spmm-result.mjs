import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { parseCsv, presentationMethodName, validateRows, readIndex, safePath } from "./spmm-data.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const input = args[0];
if (!input || input.startsWith("--")) throw new Error("Usage: add-spmm-result.mjs result.csv --source-dir PACKAGE [--root public] [--dry-run] [--merge-checkpoint]");
const root = path.resolve(option("--root", "public"));
const dryRun = args.includes("--dry-run");
const mergeCheckpoint = args.includes("--merge-checkpoint");
let text = fs.readFileSync(input, "utf8");
let rows = validateRows(parseCsv(text)).map((row) => ({ ...row, method_name: presentationMethodName(row) }));
const first = rows[0];
for (const field of ["submission_id", "method_id", "backend_id", "dataset_id"]) {
  if (!/^[A-Za-z0-9._-]+$/.test(first[field])) throw new Error("unsafe identifier: " + field);
}
const sourceDirectory = option("--source-dir", "");
if (!sourceDirectory && !dryRun) throw new Error("a corresponding source package is required");
const plugin = sourceDirectory ? JSON.parse(fs.readFileSync(path.join(sourceDirectory, "plugin.json"), "utf8")) : null;
if (plugin && (plugin.kind !== "qiwu-spmm-source-plugin" || !/^[a-f0-9]{64}$/.test(plugin.source_sha256))) throw new Error("invalid SpMM source manifest");
if (plugin && (plugin.candidate_group !== first.candidate_group || !plugin.supported_operators?.includes(first.operator_id))) {
  throw new Error("source package does not match the result adapter or operator");
}
const digest = crypto.createHash("sha256");
const payload = [];
for (const file of [...(plugin?.files || [])].sort()) {
  if (!safePath(file)) throw new Error("unsafe source file path");
  const source = path.join(sourceDirectory, "files", file);
  if (fs.lstatSync(source).isSymbolicLink()) throw new Error("source symlinks are not allowed");
  const content = fs.readFileSync(source);
  digest.update(file).update("\0").update(content).update("\0");
  payload.push({ file, content });
}
if (plugin && digest.digest("hex") !== plugin.source_sha256) throw new Error("source hash mismatch");
const relative = ["data/results/spmm", first.backend_id, first.dataset_id, first.dtype,
  "n" + first.rhs_columns, first.dense_layout, first.method_id, first.submission_id + ".csv"].join("/");
const sourceRoot = plugin ? "source/spmm/" + plugin.source_sha256 : "";
const passed = rows.filter((row) => row.status === "pass");
const metrics = {};
for (const [key, column] of [["solve-only", "solve_only_efficiency_percent"], ["pre-plus-solve", "pre_plus_solve_efficiency_percent"], ["pre-amortized", "pre_amortized_efficiency_percent"]]) {
  metrics[key] = passed.length ? passed.reduce((sum, row) => sum + Number(row[column]), 0) / passed.length : 0;
}
const entry = {
  ...Object.fromEntries(["submission_id", "method_id", "method_name", "backend_id", "hardware", "cpu_model", "dataset_id", "operator_id", "dtype", "dense_layout", "format", "base_format", "configuration_id", "candidate_group", "selection_role", "selection_metric", "ranking_scope"].map((name) => [name, first[name]])),
  rhs_columns: Number(first.rhs_columns), path: relative,
  source_manifest: sourceRoot ? sourceRoot + "/plugin.json" : "", source_sha256: plugin?.source_sha256 || "",
  csv_sha256: crypto.createHash("sha256").update(text).digest("hex"),
  public_ranked: first.public_ranked === "true", passed: passed.length,
  failed: rows.length - passed.length, total: rows.length,
  library_versions: [...new Set(rows.map((row) => row.library_version))],
  metrics,
};
const index = readIndex(root);
const previous = index.submissions.find((item) => item.submission_id === entry.submission_id);
let mergedCheckpoint = false;
if (previous && previous.csv_sha256 !== entry.csv_sha256) {
  if (!mergeCheckpoint) throw new Error("refusing to overwrite a different result with the same submission_id");
  const previousText = fs.readFileSync(path.join(root, previous.path), "utf8");
  const previousRows = validateRows(parseCsv(previousText)).map((row) => ({ ...row, method_name: presentationMethodName(row) }));
  const previousByMatrix = new Map(previousRows.map((row) => [row.matrix_id, row]));
  const mergedByMatrix = new Map(previousRows.map((row) => [row.matrix_id, row]));
  for (const row of rows) {
    const old = previousByMatrix.get(row.matrix_id);
    if (old && JSON.stringify(old) !== JSON.stringify(row)) {
      throw new Error("checkpoint overlap changed for matrix: " + row.matrix_id);
    }
    mergedByMatrix.set(row.matrix_id, row);
  }
  rows = [...mergedByMatrix.values()].sort((left, right) => left.matrix_id.localeCompare(right.matrix_id));
  const headers = text.split(/\r?\n/, 1)[0].split(",");
  const escape = (value) => {
    const cell = String(value ?? "");
    return /[",\r\n]/.test(cell) ? `"${cell.replaceAll('"', '""')}"` : cell;
  };
  text = [headers.join(","), ...rows.map((row) => headers.map((header) => escape(row[header])).join(","))].join("\n") + "\n";
  validateRows(parseCsv(text));
  mergedCheckpoint = true;
}
const finalPassed = rows.filter((row) => row.status === "pass");
entry.csv_sha256 = crypto.createHash("sha256").update(text).digest("hex");
entry.passed = finalPassed.length;
entry.failed = rows.length - finalPassed.length;
entry.total = rows.length;
entry.library_versions = [...new Set(rows.map((row) => row.library_version))];
for (const [key, column] of [["solve-only", "solve_only_efficiency_percent"], ["pre-plus-solve", "pre_plus_solve_efficiency_percent"], ["pre-amortized", "pre_amortized_efficiency_percent"]]) {
  entry.metrics[key] = finalPassed.length ? finalPassed.reduce((sum, row) => sum + Number(row[column]), 0) / finalPassed.length : 0;
}
if (!dryRun) {
  const destination = path.join(root, relative);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  if (mergedCheckpoint) fs.writeFileSync(destination, text);
  else if (!fs.existsSync(destination)) fs.writeFileSync(destination, text, { flag: "wx" });
  for (const { file, content } of payload) {
    const target = path.join(root, sourceRoot, "files", file);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    if (!fs.existsSync(target)) fs.writeFileSync(target, content, { flag: "wx" });
  }
  fs.writeFileSync(path.join(root, sourceRoot, "plugin.json"), JSON.stringify(plugin, null, 2) + "\n");
  if (previous) index.submissions[index.submissions.indexOf(previous)] = entry;
  else index.submissions.push(entry);
  index.submissions.sort((left, right) => left.path.localeCompare(right.path));
  const indexDirectory = path.join(root, "data/results/spmm");
  const indexPath = path.join(indexDirectory, "index.json");
  const temporaryIndexPath = path.join(indexDirectory, `.index-${process.pid}.tmp`);
  fs.mkdirSync(indexDirectory, { recursive: true });
  try {
    fs.writeFileSync(temporaryIndexPath, JSON.stringify(index, null, 2) + "\n");
    try {
      fs.renameSync(temporaryIndexPath, indexPath);
    } catch (error) {
      if (process.platform !== "win32" || !["EPERM", "EEXIST"].includes(error.code)) throw error;
      fs.copyFileSync(temporaryIndexPath, indexPath);
    }
  } finally {
    if (fs.existsSync(temporaryIndexPath)) fs.unlinkSync(temporaryIndexPath);
  }
}
console.log(JSON.stringify({ ...entry, rows: rows.length, target: relative, source_required: !plugin, merged_checkpoint: mergedCheckpoint }, null, 2));

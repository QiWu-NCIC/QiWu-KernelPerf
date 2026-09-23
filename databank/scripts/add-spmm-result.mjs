import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { parseCsv, validateRows, readIndex, safePath } from "./spmm-data.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const input = args[0];
if (!input || input.startsWith("--")) throw new Error("Usage: add-spmm-result.mjs result.csv --source-dir PACKAGE [--root public] [--dry-run]");
const root = path.resolve(option("--root", "public"));
const dryRun = args.includes("--dry-run");
const text = fs.readFileSync(input, "utf8");
const rows = validateRows(parseCsv(text));
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
const relative = ["data/spmm/results", first.backend_id, first.dataset_id, first.dtype,
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
if (previous && previous.csv_sha256 !== entry.csv_sha256) throw new Error("refusing to overwrite a different result with the same submission_id");
if (!dryRun) {
  const destination = path.join(root, relative);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  if (!fs.existsSync(destination)) fs.writeFileSync(destination, text, { flag: "wx" });
  for (const { file, content } of payload) {
    const target = path.join(root, sourceRoot, "files", file);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    if (!fs.existsSync(target)) fs.writeFileSync(target, content, { flag: "wx" });
  }
  fs.writeFileSync(path.join(root, sourceRoot, "plugin.json"), JSON.stringify(plugin, null, 2) + "\n");
  if (!previous) index.submissions.push(entry);
  index.submissions.sort((left, right) => left.path.localeCompare(right.path));
  fs.mkdirSync(path.join(root, "data/spmm"), { recursive: true });
  fs.writeFileSync(path.join(root, "data/spmm/index.json.tmp"), JSON.stringify(index, null, 2) + "\n");
  fs.renameSync(path.join(root, "data/spmm/index.json.tmp"), path.join(root, "data/spmm/index.json"));
}
console.log(JSON.stringify({ ...entry, rows: rows.length, target: relative, source_required: !plugin }, null, 2));

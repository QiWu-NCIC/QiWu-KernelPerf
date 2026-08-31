import fs from "node:fs";
import path from "node:path";

const [rootArg, ...inputDirs] = process.argv.slice(2);
if (!rootArg || !inputDirs.length) {
  console.error("Usage: node scripts/import-cusparse-candidates.mjs PUBLIC_ROOT INPUT_DIR...");
  process.exit(1);
}

const root = path.resolve(rootArg);
const manifestPath = path.join(root, "data", "candidate-pool", "index.json");
const candidateRoot = path.join(root, "data", "candidate-pool", "spmv");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
manifest.schema_version = 2;
manifest.result_schema = "kernelperf-spmv-v2";
const submissions = manifest.submissions || [];
const required = new Set([
  ...["coo", "csr", "csc"].flatMap((format) =>
    ["default", "alg1", "alg2"].map((algorithm) => `${format}-${algorithm}`)),
  ...[1, 2, 4, 8, 16, 32, 64, 128].flatMap((c) =>
    ["default", "alg1"].map((algorithm) => `sell-c${c}-${algorithm}`)),
  "sell-nrows-default",
  "sell-nrows-alg1",
]);

const safe = (value) => String(value || "unknown").replace(/[^A-Za-z0-9._-]+/g, "-");
const operatorOf = (value) => String(value || "spmv").split(".")[0] || "spmv";
const datasetOf = (value) => String(value || "unknown");
const canonicalMethodId = (value) => String(value || "").replace(/^cuSPARSE-CUDA-[0-9.]+-/i, "cuSPARSE-");
const fileStem = (entry) => [canonicalMethodId(entry.method_id), entry.backend_id, entry.dataset_id, entry.dtype]
  .map(safe).join("-");

function parseCsv(text) {
  const rows = [];
  let row = [], value = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { value += '"'; i += 1; }
      else quoted = !quoted;
    } else if (c === "," && !quoted) { row.push(value); value = ""; }
    else if ((c === "\n" || c === "\r") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i += 1;
      row.push(value); if (row.some((v) => v.trim())) rows.push(row); row = []; value = "";
    } else value += c;
  }
  if (value || row.length) { row.push(value); if (row.some((v) => v.trim())) rows.push(row); }
  return rows;
}

function read(file) {
  const parsed = parseCsv(fs.readFileSync(file, "utf8"));
  if (parsed.length < 2) return null;
  const headers = parsed[0];
  const rows = parsed.slice(1).map((cells) => Object.fromEntries(
    headers.map((header, i) => [header, cells[i] || ""]),
  ));
  const first = rows[0];
  if (!first || !first.method_id?.startsWith("cuSPARSE-")) return null;
  if (!required.has(first.configuration_id) || !["fp32", "fp64"].includes(first.dtype)) return null;
  const passing = rows.filter((row) => row.status === "pass" && Number(row.solve_ms) > 0);
  if (!rows.length || rows.some((row) => row.matrix_id?.startsWith("generated/"))
      || passing.length / rows.length < 0.9) return null;
  const backend = first.backend_id;
  if (!backend || !["H100-SXM5-80GB", "RTX5090-SL3061", "A100-SXM4-80GB"].includes(backend)) return null;
  return { headers, rows, first, file, mtime: fs.statSync(file).mtimeMs };
}

const csvValue = (value) => {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const serialize = (headers, rows) => [
  headers.join(","),
  ...rows.map((row) => headers.map((header) => csvValue(row[header])).join(",")),
].join("\n") + "\n";

const candidates = new Map();
for (const dir of inputDirs.map((value) => path.resolve(value))) {
  if (!fs.existsSync(dir)) continue;
  for (const file of fs.readdirSync(dir, { recursive: true })) {
    const full = path.join(dir, file);
    if (!full.toLowerCase().endsWith(".csv") || !fs.statSync(full).isFile()) continue;
    const item = read(full);
    if (!item) continue;
    const key = `${operatorOf(item.first.operator_id)}|${item.first.backend_id}|${datasetOf(item.first.dataset_id)}|${item.first.dtype}|${item.first.configuration_id}`;
    const previous = candidates.get(key);
    if (!previous || item.rows.length > previous.rows.length
        || (item.rows.length === previous.rows.length && item.mtime > previous.mtime)) {
      candidates.set(key, item);
    }
  }
}

const existing = new Map(submissions.map((entry) =>
  [`${operatorOf(entry.operator_id)}|${entry.backend_id}|${datasetOf(entry.dataset_id)}|${entry.dtype}|${entry.configuration_id}`, entry]));
for (const [key, item] of candidates) {
  const [operator, backend, dataset, dtype, configuration] = key.split("|");
  const sourceId = String(item.first.submission_id || path.basename(item.file, ".csv"));
  const fileName = `${fileStem({
    method_id: canonicalMethodId(item.first.method_id),
    backend_id: backend,
    dataset_id: dataset,
    dtype,
  })}.csv`;
  const targetDir = path.join(candidateRoot, operator, backend, dataset);
  const target = path.join(targetDir, fileName);
  fs.mkdirSync(targetDir, { recursive: true });
  const rows = item.rows.map((row) => ({
    ...row,
    method_id: canonicalMethodId(row.method_id),
    backend_id: backend,
    hardware: backend,
    dataset_id: dataset,
  }));
  fs.writeFileSync(target, serialize(item.headers, rows));
  const entry = {
    submission_id: sourceId,
    path: path.posix.join("data", "candidate-pool", operator, backend, dataset, fileName),
    method_id: canonicalMethodId(item.first.method_id),
    method_name: item.first.method_name,
    configuration_id: configuration,
    candidate_group: "cusparse",
    selection_role: "candidate",
    selected_from: "",
    base_format: item.first.base_format || "auto",
    operator_id: item.first.operator_id,
    dtype,
    backend_id: backend,
    hardware: backend,
    peak_gflops: Number(item.first.peak_gflops),
    dataset_id: item.first.dataset_id,
    source_kind: "source",
    created_at: item.first.timestamp || new Date().toISOString(),
    source_manifest: item.first.source_manifest || "source/baselines/cusparse/plugin.json",
  };
  existing.set(key, entry);
}

manifest.submissions = [...existing.values()].sort((a, b) =>
  `${operatorOf(a.operator_id)}|${a.backend_id}|${datasetOf(a.dataset_id)}|${a.dtype}|${a.configuration_id}`.localeCompare(
    `${operatorOf(b.operator_id)}|${b.backend_id}|${datasetOf(b.dataset_id)}|${b.dtype}|${b.configuration_id}`));
manifest.generated_at = new Date().toISOString();
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
const counts = {};
for (const entry of manifest.submissions) {
  if (entry.candidate_group !== "cusparse") continue;
  const key = `${operatorOf(entry.operator_id)}|${entry.backend_id}|${datasetOf(entry.dataset_id)}|${entry.dtype}`;
  counts[key] = (counts[key] || 0) + 1;
}
console.log(JSON.stringify({ imported: candidates.size, cusparse_counts: counts }, null, 2));

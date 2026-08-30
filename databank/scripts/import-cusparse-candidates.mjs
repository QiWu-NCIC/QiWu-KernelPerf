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
  if (!rows.length || rows.some((row) => row.matrix_id?.startsWith("generated/")
      || row.status !== "pass" || Number(row.solve_ms) <= 0)) return null;
  const backend = first.backend_id;
  if (!backend || !["H100-SXM5-80GB", "RTX5090-SL3061", "A100-SXM4-80GB"].includes(backend)) return null;
  return { headers, rows, first, file, mtime: fs.statSync(file).mtimeMs };
}

const candidates = new Map();
for (const dir of inputDirs.map((value) => path.resolve(value))) {
  if (!fs.existsSync(dir)) continue;
  for (const file of fs.readdirSync(dir, { recursive: true })) {
    const full = path.join(dir, file);
    if (!full.toLowerCase().endsWith(".csv") || !fs.statSync(full).isFile()) continue;
    const item = read(full);
    if (!item) continue;
    const key = `${item.first.backend_id}|${item.first.dtype}|${item.first.configuration_id}`;
    const previous = candidates.get(key);
    if (!previous || item.rows.length > previous.rows.length
        || (item.rows.length === previous.rows.length && item.mtime > previous.mtime)) {
      candidates.set(key, item);
    }
  }
}

const existing = new Map(submissions.map((entry) =>
  [`${entry.backend_id}|${entry.dtype}|${entry.configuration_id}`, entry]));
for (const [key, item] of candidates) {
  const [backend, dtype, configuration] = key.split("|");
  const sourceId = String(item.first.submission_id || path.basename(item.file, ".csv"));
  const fileName = `${sourceId}.csv`;
  const target = path.join(candidateRoot, fileName);
  fs.mkdirSync(candidateRoot, { recursive: true });
  fs.copyFileSync(item.file, target);
  const entry = {
    submission_id: sourceId,
    path: path.posix.join("data", "candidate-pool", "spmv", fileName),
    method_id: item.first.method_id,
    method_name: item.first.method_name,
    configuration_id: configuration,
    candidate_group: "cusparse",
    selection_role: "candidate",
    selected_from: "",
    base_format: item.first.base_format || "auto",
    operator_id: item.first.operator_id,
    dtype,
    backend_id: backend,
    hardware: item.first.hardware || backend,
    peak_gflops: Number(item.first.peak_gflops),
    dataset_id: item.first.dataset_id,
    source_kind: "source",
    created_at: item.first.timestamp || new Date().toISOString(),
    source_manifest: item.first.source_manifest || "source/baselines/cusparse/plugin.json",
  };
  existing.set(key, entry);
}

manifest.submissions = [...existing.values()].sort((a, b) =>
  `${a.backend_id}|${a.dtype}|${a.configuration_id}`.localeCompare(
    `${b.backend_id}|${b.dtype}|${b.configuration_id}`));
manifest.generated_at = new Date().toISOString();
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
const counts = {};
for (const entry of manifest.submissions) {
  if (entry.candidate_group !== "cusparse") continue;
  const key = `${entry.backend_id}|${entry.dtype}`;
  counts[key] = (counts[key] || 0) + 1;
}
console.log(JSON.stringify({ imported: candidates.size, cusparse_counts: counts }, null, 2));

import fs from "node:fs";
import path from "node:path";

const RESULT_COLUMNS = [
  "schema_version", "submission_id", "method_id", "method_name", "configuration_id",
  "candidate_group", "selection_role", "selected_from", "base_format",
  "operator_id", "dtype", "backend_id", "hardware", "peak_gflops", "job_id",
  "dataset_id", "matrix_id", "matrix_name", "rows", "cols", "nnz", "status", "operations",
  "preprocess_ms", "solve_ms", "solve_gflops", "solve_only_efficiency_percent",
  "timestamp", "source_kind",
];
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const args = process.argv.slice(2);
const inputPath = args[0] && !args[0].startsWith("--") ? args[0] : "";
const requestedSubmissionId = option("--submission-id");
const requestedBackendId = option("--backend");
const sourceDir = option("--source-dir");
const dryRun = args.includes("--dry-run");
const indexPath = "public/data/index.json";
const safe = (value) => String(value || "unknown").replace(/[^A-Za-z0-9._-]+/g, "-");

if (!inputPath) {
  console.error(
    "Usage: npm run add:spmv -- result.csv [--backend a100-server] " +
    "[--submission-id ID] [--source-dir DIR] [--dry-run]",
  );
  process.exit(1);
}

const csv = fs.readFileSync(inputPath, "utf8");
const parsed = parseCsv(csv);
if (parsed.length < 2) {
  throw new Error("result CSV must contain a header and at least one row");
}
const headers = parsed[0];
const OPTIONAL_RESULT_COLUMNS = [
  "configuration_id", "candidate_group", "selection_role", "selected_from",
];
const missing = RESULT_COLUMNS.filter(
  (header) => !headers.includes(header) && !OPTIONAL_RESULT_COLUMNS.includes(header),
);
if (missing.length) {
  throw new Error("missing required result column(s): " + missing.join(", "));
}
const rows = parsed.slice(1).map((cells) =>
  Object.fromEntries(headers.map((header, index) => [header, cells[index] || ""]))
);
const first = rows[0];
const submissionId = requestedSubmissionId || first.submission_id;
const backendId = requestedBackendId || first.backend_id;
if (!ID_PATTERN.test(submissionId)) {
  throw new Error("submission_id must be a filesystem-safe identifier");
}
if (!ID_PATTERN.test(backendId)) {
  throw new Error("backend_id must be a filesystem-safe identifier");
}
if (requestedSubmissionId && requestedSubmissionId !== first.submission_id) {
  throw new Error("--submission-id must match the CSV submission_id");
}
if (requestedBackendId && requestedBackendId !== first.backend_id) {
  throw new Error("--backend must match the CSV backend_id");
}

const identityColumns = [
  "submission_id", "method_id", "method_name", "configuration_id", "candidate_group",
  "selection_role", "selected_from", "base_format", "operator_id",
  "dtype", "backend_id", "hardware", "peak_gflops", "job_id", "dataset_id",
  "source_kind",
];
const matrixIds = new Set();
rows.forEach((row, index) => {
  const line = index + 2;
  identityColumns.filter((column) => headers.includes(column)).forEach((column) => {
    if (row[column] !== first[column]) {
      throw new Error("line " + line + " has inconsistent " + column);
    }
  });
  if (row.schema_version !== "2") {
    throw new Error("line " + line + " uses unsupported schema_version");
  }
  if (!["fp32", "fp64"].includes(row.dtype)) {
    throw new Error("line " + line + " has unsupported dtype");
  }
  if (!["pass", "error", "fail"].includes(row.status)) {
    throw new Error("line " + line + " has unsupported status");
  }
  if (!["source", "object", "derived"].includes(row.source_kind)) {
    throw new Error("line " + line + " has unsupported source_kind");
  }
  if (!row.matrix_id || matrixIds.has(row.matrix_id)) {
    throw new Error("line " + line + " has an empty or duplicate matrix_id");
  }
  matrixIds.add(row.matrix_id);
  ["rows", "cols", "nnz", "operations", "preprocess_ms", "solve_ms", "peak_gflops"]
    .forEach((column) => {
      if (!Number.isFinite(Number(row[column]))) {
        throw new Error("line " + line + " has invalid " + column);
      }
    });
  if (Number(row.peak_gflops) <= 0 || Number(row.operations) <= 0) {
    throw new Error("line " + line + " must have positive peak_gflops and operations");
  }
  if (Number(row.preprocess_ms) < 0 || Number(row.solve_ms) < 0) {
    throw new Error("line " + line + " has a negative timing");
  }
  if (row.status === "pass" && Number(row.solve_ms) <= 0) {
    throw new Error("line " + line + " has a non-positive solve_ms");
  }
});

const operatorId = String(first.operator_id || "spmv").split(".")[0] || "spmv";
const datasetId = first.dataset_id || "unknown";
const fileName = [first.method_id, backendId, datasetId, first.dtype]
  .map(safe).join("-") + ".csv";
const outputDir = path.join("public/data/results", operatorId, backendId, datasetId);
const target = path.join(outputDir, fileName);
let sourcePackage = null;
let sourceTarget = null;
const manifest = JSON.parse(fs.readFileSync(indexPath, "utf8"));
const entry = {
  submission_id: submissionId,
  path: path.relative("public", target).replaceAll("\\", "/"),
  method_id: first.method_id,
  method_name: first.method_name,
  configuration_id: first.configuration_id,
  candidate_group: first.candidate_group,
  selection_role: first.selection_role,
  selected_from: first.selected_from,
  base_format: first.base_format,
  operator_id: first.operator_id,
  dtype: first.dtype,
  backend_id: first.backend_id,
  hardware: first.hardware,
  peak_gflops: Number(first.peak_gflops),
  dataset_id: first.dataset_id,
  source_kind: first.source_kind,
  created_at: first.timestamp || new Date().toISOString(),
};
if (sourceDir) {
  if (!fs.statSync(sourceDir, { throwIfNoEntry: false })?.isDirectory()) {
    throw new Error("--source-dir must be a standalone plugin package directory");
  }
  const sourceManifestPath = path.join(sourceDir, "plugin.json");
  if (!fs.existsSync(sourceManifestPath)) {
    throw new Error("--source-dir must contain plugin.json");
  }
  sourcePackage = JSON.parse(fs.readFileSync(sourceManifestPath, "utf8"));
  if (sourcePackage.kind !== "qiwu-spmv-source-plugin"
      || !/^[0-9a-f]{64}$/.test(sourcePackage.source_sha256 || "")) {
    throw new Error("--source-dir is not a Qiwu standalone source plugin package");
  }
  if (!Array.isArray(sourcePackage.files) || !sourcePackage.files.length) {
    throw new Error("--source-dir manifest must list plugin files");
  }
  for (const relative of sourcePackage.files) {
    const parts = String(relative).split("/");
    if (!relative || String(relative).includes("\\")
        || parts.some((part) => !part || part === "." || part === "..")) {
      throw new Error("--source-dir manifest contains an unsafe plugin path");
    }
    if (!fs.existsSync(path.join(sourceDir, "files", relative))) {
      throw new Error("--source-dir is missing plugin file: " + relative);
    }
  }
  sourceTarget = path.join("public/source", sourcePackage.source_sha256);
  entry.source_sha256 = sourcePackage.source_sha256;
  entry.source_manifest = path.posix.join("source", sourcePackage.source_sha256, "plugin.json");
}
const normalizedCsv = serializeCsv(headers, rows);
const existingEntry = (manifest.submissions || []).find(
  (item) => item.submission_id === submissionId,
);
const unchanged = Boolean(
  existingEntry
  && fs.existsSync(target)
  && fs.readFileSync(target, "utf8") === normalizedCsv
  && Object.entries(entry).every(([key, value]) => existingEntry[key] === value)
  && (!sourceDir || fs.existsSync(path.join(sourceTarget, "plugin.json")))
);
if (!unchanged) {
  manifest.submissions = (manifest.submissions || []).filter(
    (item) => item.submission_id !== submissionId,
  );
  manifest.submissions.push(entry);
  manifest.generated_at = new Date().toISOString();
}

const summary = {
  dry_run: dryRun,
  source: inputPath,
  target,
  rows: rows.length,
  changed: !unchanged,
  entry,
  source_dir: sourceDir || null,
};
if (!dryRun && !unchanged) {
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(target, normalizedCsv);
  if (existingEntry?.path && existingEntry.path !== path.relative("public", target).replaceAll("\\", "/")) {
    const previous = path.join("public", existingEntry.path.replaceAll("/", path.sep));
    if (fs.existsSync(previous)) fs.rmSync(previous);
  }
  if (sourceDir && !fs.existsSync(sourceTarget)) {
    fs.cpSync(sourceDir, sourceTarget, { recursive: true });
  }
  fs.writeFileSync(indexPath, JSON.stringify(manifest, null, 2) + "\n");
}
console.log(JSON.stringify(summary, null, 2));

function option(name) {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  if (!args[index + 1] || args[index + 1].startsWith("--")) {
    throw new Error(name + " requires a value");
  }
  return args[index + 1];
}

function parseCsv(text) {
  const result = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(value);
      if (row.some((cell) => cell.trim())) result.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }
  if (quoted) throw new Error("CSV contains an unterminated quoted field");
  if (value || row.length) {
    row.push(value);
    if (row.some((cell) => cell.trim())) result.push(row);
  }
  return result;
}

function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function serializeCsv(headers, rows) {
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvValue(row[header])).join(",")),
  ].join("\n") + "\n";
}

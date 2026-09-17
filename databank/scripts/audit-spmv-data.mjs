import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const root = path.resolve(process.argv[2] || "public");
const failures = [];
const manifest = JSON.parse(fs.readFileSync(path.join(root, "data", "index.json"), "utf8"));
if (manifest.schema_version !== 2 || manifest.result_schema !== "kernelperf-spmv-v2") {
  failures.push("manifest schema is not kernelperf-spmv-v2");
}
const candidateManifest = JSON.parse(fs.readFileSync(
  path.join(root, "data", "candidate-pool", "index.json"), "utf8",
));
if (candidateManifest.schema_version !== 2
    || candidateManifest.result_schema !== "kernelperf-spmv-v2") {
  failures.push("candidate manifest schema is not kernelperf-spmv-v2");
}
const required = new Set([
  ...["coo", "csr", "csc"].flatMap((format) =>
    ["default", "alg1", "alg2"].map((algorithm) => `${format}-${algorithm}`)),
  ...[1, 2, 4, 8, 16, 32, 64, 128].flatMap((c) =>
    ["default", "alg1"].map((algorithm) => `sell-c${c}-${algorithm}`)),
  "sell-nrows-default", "sell-nrows-alg1",
]);
const canonicalBackend = (value) => /ict-a100/i.test(String(value || ""))
  ? "A100-SXM4-80GB" : value;
const datasetOf = (value) => String(value || "unknown");
const operatorOf = (value) => String(value || "spmv").split(".")[0] || "spmv";
const publicRoot = root;
function walkCsv(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((item) => {
    const file = path.join(directory, item.name);
    if (item.isDirectory()) return walkCsv(file);
    return item.name.endsWith(".csv") ? [file] : [];
  });
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
function readCsv(file) {
  const [headers = [], ...rows] = parseCsv(fs.readFileSync(file, "utf8"));
  return { headers, rows };
}
const publicKeys = new Set();
const publicPaths = new Set();
const sourceManifests = new Set();
for (const entry of manifest.submissions || []) {
  const key = `${operatorOf(entry.operator_id)}|${entry.method_id}|${entry.backend_id}|${entry.dataset_id}|${entry.dtype}`;
  if (publicKeys.has(key)) failures.push(`duplicate public key: ${key}`);
  publicKeys.add(key);
  const file = path.join(publicRoot, entry.path.replaceAll("/", path.sep));
  publicPaths.add(path.resolve(file));
  if (!fs.existsSync(file)) failures.push(`missing public CSV: ${entry.path}`);
  const expectedPrefix = `data/results/${operatorOf(entry.operator_id)}/${canonicalBackend(entry.backend_id)}/${datasetOf(entry.dataset_id)}/`;
  if (!entry.path.startsWith(expectedPrefix)) failures.push(`public CSV is outside dataset scope: ${entry.path}`);
  if (/smoke/i.test(entry.path) || /generated\//i.test(entry.path)) failures.push(`temporary public entry: ${entry.path}`);
  if (!entry.source_manifest) failures.push(`missing source manifest: ${key}`);
  else sourceManifests.add(entry.source_manifest);
  if (/^alphasparselib-csr-best$/i.test(entry.method_id) && entry.base_format !== "manual-selection") {
    failures.push(`AlphaSparseLib BEST base format is not manual-selection: ${entry.path}`);
  }
}
const requiredPluginFiles = [
  "CMakeLists.txt",
  "README-QIWU-PLUGIN.md",
  "include/qiwu/spmv_plugin.cuh",
  "examples/standalone.cu",
];
for (const relativeManifest of sourceManifests) {
  const manifestFile = path.join(publicRoot, relativeManifest.replaceAll("/", path.sep));
  if (!fs.existsSync(manifestFile)) {
    failures.push(`missing source manifest file: ${relativeManifest}`);
    continue;
  }
  const source = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
  if (source.kind !== "qiwu-spmv-source-plugin") {
    failures.push(`source is not a standalone plugin: ${relativeManifest}`);
  }
  if (!/^[0-9a-f]{64}$/.test(source.source_sha256 || "")) {
    failures.push(`source has no content hash: ${relativeManifest}`);
  }
  const files = new Set(source.files || []);
  for (const configuration of source.configurations || []) {
    if (!configuration.configuration_id || !files.has(configuration.entry_source)) {
      failures.push(`invalid source configuration: ${relativeManifest}`);
    }
  }
  for (const requiredFile of requiredPluginFiles) {
    if (!files.has(requiredFile)) failures.push(`source package lacks ${requiredFile}: ${relativeManifest}`);
  }
  const sourceRoot = path.dirname(manifestFile);
  const digest = crypto.createHash("sha256");
  for (const relative of [...files].sort()) {
    const parts = String(relative).split("/");
    if (!relative || String(relative).includes("\\")
        || parts.some((part) => !part || part === "." || part === "..")) {
      failures.push(`unsafe source path: ${relativeManifest}/${relative}`);
      continue;
    }
    const file = path.join(sourceRoot, "files", relative.replaceAll("/", path.sep));
    if (!fs.existsSync(file)) failures.push(`missing source file: ${relativeManifest}/${relative}`);
    else {
      const content = fs.readFileSync(file);
      digest.update(relative);
      digest.update("\0");
      digest.update(content);
      digest.update("\0");
      if (/\.(?:cu|cuh|h|hpp)$/.test(relative)
          && /kernelperf_spmv_|KERNELPERF_SPMV_FP64/.test(content.toString("utf8"))) {
        failures.push(`source package uses removed contract: ${relativeManifest}/${relative}`);
      }
    }
  }
  if (digest.digest("hex") !== source.source_sha256) {
    failures.push(`source content hash mismatch: ${relativeManifest}`);
  }
}
const groups = new Map();
const candidateMinimums = new Map();
const candidatePaths = new Set();
for (const entry of candidateManifest.submissions || []) {
  const file = path.join(publicRoot, entry.path.replaceAll("/", path.sep));
  candidatePaths.add(path.resolve(file));
  if (entry.candidate_group !== "cusparse" || !required.has(entry.configuration_id)) continue;
  if (!fs.existsSync(file)) { failures.push(`missing candidate CSV: ${entry.path}`); continue; }
  const { headers, rows } = readCsv(file);
  const schemaIndex = headers.indexOf("schema_version");
  if (!headers.includes("solve_only_efficiency_percent")) {
    failures.push(`missing solve-only efficiency column: ${entry.path}`);
  }
  if (rows.some((row) => row[schemaIndex] !== "2")) failures.push(`invalid schema row: ${entry.path}`);
  const matrixIndex = headers.indexOf("matrix_id");
  const statuses = headers.indexOf("status");
  const solveIndex = headers.indexOf("solve_ms");
  const passing = rows.filter((row) => row[statuses] === "pass" && Number(row[solveIndex]) > 0);
  // Coverage is a ranking policy, not a CSV validity requirement. A complete
  // 100-matrix sweep may contain failed cases; its passing rows still belong
  // in per-matrix BEST selection, while the frontend applies the rankability
  // threshold when displaying the aggregate result.
  if (rows.length !== 100 || rows.some((row) => row[matrixIndex]?.startsWith("generated/"))) {
    failures.push(`incomplete candidate: ${entry.backend_id}/${entry.dtype}/${entry.configuration_id}`);
    continue;
  }
  const key = `${operatorOf(entry.operator_id)}|${canonicalBackend(entry.backend_id)}|${datasetOf(entry.dataset_id)}|${entry.dtype}`;
  const expectedPrefix = `data/candidate-pool/${operatorOf(entry.operator_id)}/${canonicalBackend(entry.backend_id)}/${datasetOf(entry.dataset_id)}/`;
  if (!entry.path.startsWith(expectedPrefix)) failures.push(`candidate CSV is outside dataset scope: ${entry.path}`);
  const ids = rows.map((row) => row[matrixIndex]).sort();
  const minimums = candidateMinimums.get(key) || new Map();
  for (const row of rows) {
    const matrix = row[matrixIndex];
    if (row[statuses] !== "pass" || Number(row[solveIndex]) <= 0) continue;
    const solve = Number(row[solveIndex]);
    if (!minimums.has(matrix) || solve < minimums.get(matrix)) minimums.set(matrix, solve);
  }
  candidateMinimums.set(key, minimums);
  const previous = groups.get(key);
  if (previous && previous.join("\n") !== ids.join("\n")) {
    failures.push(`matrix-set mismatch: ${key}/${entry.configuration_id}`);
  } else if (!previous) groups.set(key, ids);
}
for (const entry of manifest.submissions || []) {
  if (entry.method_id !== "cusparse-best") continue;
  const file = path.join(publicRoot, entry.path.replaceAll("/", path.sep));
  if (!fs.existsSync(file)) continue;
  const { headers, rows } = readCsv(file);
  const schemaIndex = headers.indexOf("schema_version");
  const baseFormatIndex = headers.indexOf("base_format");
  if (!headers.includes("solve_only_efficiency_percent")) {
    failures.push(`missing solve-only efficiency column: ${entry.path}`);
  }
  if (entry.base_format !== "manual-selection") {
    failures.push(`cuSPARSE BEST base format is not manual-selection: ${entry.path}`);
  }
  const matrixIndex = headers.indexOf("matrix_id");
  const solveIndex = headers.indexOf("solve_ms");
  const minimums = candidateMinimums.get(`${operatorOf(entry.operator_id)}|${canonicalBackend(entry.backend_id)}|${datasetOf(entry.dataset_id)}|${entry.dtype}`);
  for (const row of rows) {
    if (row[schemaIndex] !== "2") failures.push(`invalid schema row: ${entry.path}`);
    if (row[baseFormatIndex] !== "manual-selection") {
      failures.push(`cuSPARSE BEST row base format is not manual-selection: ${entry.path}`);
    }
    const expected = minimums?.get(row[matrixIndex]);
    if (expected === undefined || Math.abs(Number(row[solveIndex]) - expected) > 1e-9) {
      failures.push(`BEST is not candidate minimum: ${entry.backend_id}/${entry.dtype}/${row[matrixIndex]}`);
    }
  }
}
for (const entry of manifest.submissions || []) {
  if (!/^alphasparselib-csr-best$/i.test(entry.method_id)) continue;
  const file = path.join(publicRoot, entry.path.replaceAll("/", path.sep));
  if (!fs.existsSync(file)) continue;
  const { headers, rows } = readCsv(file);
  const baseFormatIndex = headers.indexOf("base_format");
  if (rows.some((row) => row[baseFormatIndex] !== "manual-selection")) {
    failures.push(`AlphaSparseLib BEST row base format is not manual-selection: ${entry.path}`);
  }
}
const observedCandidateScopes = new Map();
for (const entry of candidateManifest.submissions || []) {
  if (entry.candidate_group !== "cusparse") continue;
  const key = `${operatorOf(entry.operator_id)}|${canonicalBackend(entry.backend_id)}|${datasetOf(entry.dataset_id)}|${entry.dtype}`;
  if (!observedCandidateScopes.has(key)) observedCandidateScopes.set(key, new Set());
  observedCandidateScopes.get(key).add(entry.configuration_id);
}
for (const [key, configurations] of observedCandidateScopes) {
  if ([...configurations].some((value) => !required.has(value))) {
    failures.push(`unknown candidate configuration: ${key}`);
  }
  const [operator, backend, dataset, dtype] = key.split("|");
  const publishesBest = (manifest.submissions || []).some((entry) =>
    entry.method_id === "cusparse-best" && operatorOf(entry.operator_id) === operator
      && canonicalBackend(entry.backend_id) === backend && datasetOf(entry.dataset_id) === dataset
      && entry.dtype === dtype,
  );
  if (publishesBest && (required.size !== configurations.size
      || [...required].some((value) => !configurations.has(value)))) {
    failures.push(`BEST uses incomplete candidate scope: ${key} (${configurations.size}/${required.size})`);
  }
}
for (const file of walkCsv(path.join(publicRoot, "data", "results"))) {
  if (!publicPaths.has(path.resolve(file))) failures.push(`unindexed public CSV: ${file}`);
}
for (const file of walkCsv(path.join(publicRoot, "data", "candidate-pool"))) {
  if (!candidatePaths.has(path.resolve(file))) failures.push(`unindexed candidate CSV: ${file}`);
}
const result = {
  public_submissions: (manifest.submissions || []).length,
  public_by_backend: (manifest.submissions || []).reduce((counts, entry) => {
    counts[entry.backend_id] = (counts[entry.backend_id] || 0) + 1;
    return counts;
  }, {}),
  cusparse_candidates_per_backend_dtype: required.size,
  failures,
};
console.log(JSON.stringify(result, null, 2));
if (failures.length) process.exitCode = 1;

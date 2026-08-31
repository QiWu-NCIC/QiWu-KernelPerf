import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] || "public");
const dataRoot = path.join(root, "data");
const resultRoot = path.join(dataRoot, "results", "spmv");
const manifestPath = path.join(dataRoot, "index.json");
const candidateManifestPath = path.join(dataRoot, "candidate-pool", "index.json");
const canonicalBackend = "A100-SXM4-80GB";
const canonicalHardware = "A100-SXM4-80GB";
const requiredGhostConfigurations = Array.from(
  { length: 18 },
  (_, exponent) => "sigma-" + (2 ** exponent),
);
const publicCusparseConfigurations = new Set([
  ...["coo", "csr", "csc"].flatMap((format) =>
    ["default", "alg1", "alg2"].map((algorithm) => `${format}-${algorithm}`)),
  "sell-nrows-default",
  "sell-nrows-alg1",
]);

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
manifest.schema_version = 2;
manifest.result_schema = "kernelperf-spmv-v2";
const sourceEntries = manifest.submissions || [];
const candidateEntries = fs.existsSync(candidateManifestPath)
  ? JSON.parse(fs.readFileSync(candidateManifestPath, "utf8")).submissions || []
  : [];
function platformOf(value) {
  const text = `${value.backend_id || ""} ${value.hardware || ""} ${value.submission_id || ""}`;
  if (/H100-SXM5-80GB/i.test(text)) return "H100-SXM5-80GB";
  if (/RTX5090-SL3061/i.test(text)) return "RTX5090-SL3061";
  if (/ict-a100/i.test(text)) return canonicalBackend;
  return value.backend_id || canonicalBackend;
}

function operatorOf(value) {
  return String(value.operator_id || "spmv").split(".")[0] || "spmv";
}

function datasetOf(value) {
  return String(value.dataset_id || "unknown");
}

function canonicalMethodId(value) {
  return String(value || "").replace(/^cuSPARSE-CUDA-[0-9.]+-/i, "cuSPARSE-");
}

function cuSparseName(backend, suffix) {
  const version = backend === canonicalBackend ? "12.9" : "12.8";
  return `cuSPARSE CUDA ${version} ${suffix}`;
}

function sourceManifestFor(entry) {
  if (entry.source_manifest) return entry.source_manifest;
  const method = String(entry.method_id || "").toLowerCase();
  const group = String(entry.candidate_group || "").toLowerCase();
  if (method.includes("cusparse") || group === "cusparse") return "source/baselines/cusparse/plugin.json";
  if (method.includes("ghost") || group.includes("ghost")) return "source/baselines/ghost-sell/plugin.json";
  if (method.includes("csr5")) return "source/baselines/csr5/plugin.json";
  if (method.includes("adaptive") || group.includes("adaptive")) return "source/baselines/csr-adaptive/plugin.json";
  if (method.includes("alphasparse") || group.includes("alphasparse")) return "source/baselines/alphasparse/plugin.json";
  return "";
}

// Prefer the curated candidate-pool file over an older public copy with the
// same logical operator, platform, dataset, dtype, method, and configuration.
const candidateKeys = new Set(candidateEntries.map((entry) =>
  `${operatorOf(entry)}|${platformOf(entry)}|${datasetOf(entry)}|${entry.dtype || ""}|${entry.candidate_group || ""}|${entry.method_id || ""}|${entry.configuration_id || ""}`));
for (const entry of sourceEntries) {
  const key = `${operatorOf(entry)}|${platformOf(entry)}|${datasetOf(entry)}|${entry.dtype || ""}|${entry.candidate_group || ""}|${entry.method_id || ""}|${entry.configuration_id || ""}`;
  if (!candidateKeys.has(key)) {
    candidateEntries.push(entry);
    candidateKeys.add(key);
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (c === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((c === "\n" || c === "\r") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i += 1;
      row.push(value);
      if (row.some((item) => item.trim() !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += c;
    }
  }
  if (quoted) throw new Error("unterminated CSV quote");
  if (value || row.length) {
    row.push(value);
    if (row.some((item) => item.trim() !== "")) rows.push(row);
  }
  return rows;
}

function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text)
    ? '"' + text.replaceAll('"', '""') + '"'
    : text;
}

function readEntry(entry) {
  const file = path.join(root, entry.path.replaceAll("/", path.sep));
  if (!fs.existsSync(file)) return null;
  const parsed = parseCsv(fs.readFileSync(file, "utf8"));
  if (parsed.length < 2) return null;
  const headers = parsed[0];
  if (!headers.includes("solve_only_efficiency_percent")) {
    throw new Error(`${entry.path}: expected kernelperf-spmv-v2 CSV`);
  }
  const rows = parsed.slice(1).map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index] || ""])),
  );
  if (rows.some((row) => row.schema_version !== "2")) {
    throw new Error(`${entry.path}: schema_version must be 2`);
  }
  return { entry, headers, rows };
}

function normalizeRow(row) {
  row.schema_version = "2";
  row.backend_id = platformOf(row);
  row.hardware = row.hardware
    ? platformOf({ backend_id: row.hardware, hardware: row.hardware })
    : row.backend_id;
  row.method_id = canonicalMethodId(row.method_id);
  if (row.method_id?.startsWith("AlphaSparse-")) {
    row.method_id = row.method_id.replace(/^AlphaSparse-/, "AlphaSparseLib-");
  }
  if (row.method_name?.startsWith("AlphaSparse ")) {
    row.method_name = row.method_name.replace(/^AlphaSparse /, "AlphaSparseLib ");
  }
  if (row.method_id === "alphasparse-csr-best") {
    row.method_id = "AlphaSparseLib-CSR-BEST";
    row.method_name = "AlphaSparseLib CSR BEST";
  }
  if (row.method_id === "AlphaSparseLib-CSR-best") {
    row.method_id = "AlphaSparseLib-CSR-BEST";
    row.method_name = "AlphaSparseLib CSR BEST";
  }
  if (row.candidate_group === "alphasparse-csr") {
    row.candidate_group = "AlphaSparseLib-CSR";
  }
  if (row.candidate_group === "GHOST-SELL-C32-sigma") {
    row.candidate_group = "ghost-sell";
  }
  if (row.method_id?.startsWith("GHOST-SELL-C32-sigma-")) {
    row.method_id = row.method_id.toLowerCase();
  }
  return row;
}

function normalizeEntry(entry) {
  const normalized = { ...entry };
  const submissionId = String(normalized.submission_id || "");
  normalized.submission_id = submissionId.toUpperCase().includes(canonicalBackend)
    ? submissionId
    : submissionId.replace(/ict-a100(?:-gpu\d+|-opencl)?/gi, canonicalBackend);
  normalized.backend_id = platformOf(normalized);
  normalized.hardware = normalized.hardware
    ? platformOf({ backend_id: normalized.hardware, hardware: normalized.hardware })
    : normalized.backend_id;
  normalized.method_id = canonicalMethodId(normalized.method_id);
  normalized.source_manifest = sourceManifestFor(normalized);
  if (normalized.method_id?.startsWith("AlphaSparse-")) {
    normalized.method_id = normalized.method_id.replace(/^AlphaSparse-/, "AlphaSparseLib-");
  }
  if (normalized.method_name?.startsWith("AlphaSparse ")) {
    normalized.method_name = normalized.method_name.replace(/^AlphaSparse /, "AlphaSparseLib ");
  }
  if (normalized.method_id === "alphasparse-csr-best") {
    normalized.method_id = "AlphaSparseLib-CSR-BEST";
    normalized.method_name = "AlphaSparseLib CSR BEST";
  }
  if (normalized.method_id === "AlphaSparseLib-CSR-best") {
    normalized.method_id = "AlphaSparseLib-CSR-BEST";
    normalized.method_name = "AlphaSparseLib CSR BEST";
  }
  if (normalized.candidate_group === "alphasparse-csr") {
    normalized.candidate_group = "AlphaSparseLib-CSR";
  }
  if (normalized.candidate_group === "GHOST-SELL-C32-sigma") {
    normalized.candidate_group = "ghost-sell";
  }
  if (normalized.method_id?.startsWith("GHOST-SELL-C32-sigma-")) {
    normalized.method_id = normalized.method_id.toLowerCase();
  }
  return normalized;
}

function isSellCandidate(entry) {
  return /^cuSPARSE-SELL-C(?:1|2|4|8|16|32|64|128)-(ALG1|DEFAULT)$/.test(entry.method_id || "");
}

function sellKey(entry) {
  const match = /^cuSPARSE-SELL-C(?:1|2|4|8|16|32|64|128)-(ALG1|DEFAULT)$/.exec(entry.method_id);
  return match ? match[1] : null;
}

function sellC(entry) {
  const match = /^cuSPARSE-SELL-C(\d+)-(?:ALG1|DEFAULT)$/.exec(entry.method_id);
  return match ? Number(match[1]) : null;
}

function sellSelectionKey(entry) {
  return `${operatorOf(entry)}:${platformOf(entry)}:${datasetOf(entry)}:${entry.dtype}:${entry.configuration_id}`;
}

function geometricMean(rows) {
  const values = rows
    .filter((row) => row.status === "pass" && Number(row.solve_gflops) > 0)
    .map((row) => Math.log(Number(row.solve_gflops)));
  return values.length
    ? Math.exp(values.reduce((sum, value) => sum + value, 0) / values.length)
    : 0;
}

function hasCoverage(rows, threshold = 0.9) {
  if (!rows.length) return false;
  const passing = rows.filter((row) => row.status === "pass" && Number(row.solve_ms) > 0);
  return passing.length / rows.length >= threshold;
}

function writeCsv(file, headers, rows) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const output = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvValue(row[header])).join(",")),
  ].join("\n") + "\n";
  fs.writeFileSync(file, output);
}

function resultFileStem(entry) {
  return [entry.method_id, entry.backend_id, entry.dataset_id, entry.dtype]
    .map((value) => String(value || "unknown").replace(/[^A-Za-z0-9._-]+/g, "-"))
    .join("-");
}

const loaded = sourceEntries.map(readEntry).filter(Boolean);
const candidateLoaded = candidateEntries.length
  ? candidateEntries.map(readEntry).filter(Boolean)
  : loaded;
const selectedSell = new Set();
for (const algorithm of ["ALG1", "DEFAULT"]) {
  const candidates = candidateLoaded.filter(({ entry }) =>
    isSellCandidate(entry) && sellKey(entry) === algorithm);
  const backends = [...new Set(candidates.map(({ entry }) => platformOf(entry)))];
  for (const backend of backends) {
    const byC = new Map();
    for (const candidate of candidates.filter(({ entry }) => platformOf(entry) === backend)) {
      const c = sellC(candidate.entry);
      if (!byC.has(c)) byC.set(c, []);
      byC.get(c).push(candidate);
    }
    const ranked = [...byC.entries()]
      .filter(([, group]) => new Set(group.map(({ entry }) => entry.dtype)).size === 2)
      .map(([c, group]) => ({ c, group, score: geometricMean(group.flatMap(({ rows }) => rows)) }))
      .sort((a, b) => b.score - a.score || a.c - b.c);
    for (const candidate of ranked[0]?.group || []) {
      selectedSell.add(sellSelectionKey(candidate.entry));
    }
  }
}

// Keep one concrete GHOST configuration per dtype/platform in the public
// contest.  The complete sweep remains available in candidate-pool for
// reproducibility, while the public view shows the geometric-mean winner.
const ghostCandidateLoaded = candidateLoaded
  .map(({ entry, headers, rows }) => ({
    entry: normalizeEntry(entry),
    headers,
    rows: rows.map((row) => normalizeRow({ ...row })),
  }))
  .filter(({ entry }) => (entry.method_id?.startsWith("ghost-sell-c32-sigma-")
      || entry.method_id?.startsWith("GHOST-SELL-C32-sigma-"))
    && entry.selection_role !== "best")
  .filter(({ rows }) => hasCoverage(rows));
const selectedGhost = new Set();
const ghostScopes = new Set(ghostCandidateLoaded.map(({ entry }) =>
  `${operatorOf(entry)}:${entry.backend_id || canonicalBackend}:${datasetOf(entry)}`));
for (const scope of ghostScopes) {
  const [operator, backend, dataset] = scope.split(":");
  const byConfiguration = new Map();
  for (const candidate of ghostCandidateLoaded.filter(({ entry }) =>
    operatorOf(entry) === operator && (entry.backend_id || canonicalBackend) === backend
      && datasetOf(entry) === dataset)) {
    const key = candidate.entry.configuration_id;
    if (!byConfiguration.has(key)) byConfiguration.set(key, []);
    byConfiguration.get(key).push(candidate);
  }
  const winner = [...byConfiguration.entries()]
    .filter(([, group]) => new Set(group.map(({ entry }) => entry.dtype)).size === 2)
    .map(([configuration, group]) => ({
      configuration,
      group,
      score: geometricMean(group.flatMap(({ rows }) => rows)),
    }))
    .sort((a, b) => b.score - a.score
      || String(a.configuration).localeCompare(String(b.configuration)))[0];
  if (winner) {
    for (const candidate of winner.group) {
      selectedGhost.add(`${operator}:${backend}:${dataset}:${candidate.entry.dtype}:${winner.configuration}`);
    }
  }
}

const retained = loaded.filter(({ entry }) => {
  if (entry.candidate_group === "baseline-regression") return true;
  if (["alphasparse-csr", "AlphaSparseLib-CSR"].includes(entry.candidate_group)) return true;
  if (isSellCandidate(entry)) return selectedSell.has(sellSelectionKey(entry));
  if (["ghost-sell", "GHOST-SELL-C32-sigma"].includes(entry.candidate_group)
    && entry.selection_role !== "best") {
    return selectedGhost.has(`${operatorOf(entry)}:${platformOf(entry)}:${datasetOf(entry)}:${entry.dtype}:${entry.configuration_id}`);
  }
  return entry.candidate_group === "cusparse"
    && entry.selection_role !== "best";
});
const retainedCusparseKeys = new Set(retained
  .filter(({ entry }) => entry.candidate_group === "cusparse")
  .map(({ entry }) => `${operatorOf(entry)}:${platformOf(entry)}:${datasetOf(entry)}:${entry.dtype}:${entry.configuration_id}`));
for (const candidate of candidateLoaded) {
  const entry = candidate.entry;
  if (entry.candidate_group !== "cusparse"
      || entry.selection_role === "best"
      || !publicCusparseConfigurations.has(entry.configuration_id)
      || !hasCoverage(candidate.rows)) continue;
  const key = `${operatorOf(entry)}:${platformOf(entry)}:${datasetOf(entry)}:${entry.dtype}:${entry.configuration_id}`;
  if (!retainedCusparseKeys.has(key)) {
    retained.push(candidate);
    retainedCusparseKeys.add(key);
  }
}
const retainedSellKeys = new Set(
  retained.filter(({ entry }) => isSellCandidate(entry)).map(({ entry }) => sellSelectionKey(entry)),
);
for (const candidate of candidateLoaded) {
  const key = sellSelectionKey(candidate.entry);
  if (isSellCandidate(candidate.entry) && selectedSell.has(key) && !retainedSellKeys.has(key)) {
    retained.push(candidate);
    retainedSellKeys.add(key);
  }
}
const retainedGhostKeys = new Set(retained
  .filter(({ entry }) => ["ghost-sell", "GHOST-SELL-C32-sigma"].includes(entry.candidate_group))
  .map(({ entry }) => `${operatorOf(entry)}:${platformOf(entry)}:${datasetOf(entry)}:${entry.dtype}:${entry.configuration_id}`));
for (const candidate of ghostCandidateLoaded) {
  const key = `${operatorOf(candidate.entry)}:${candidate.entry.backend_id}:${datasetOf(candidate.entry)}:${candidate.entry.dtype}:${candidate.entry.configuration_id}`;
  if (selectedGhost.has(key) && !retainedGhostKeys.has(key)) {
    retained.push(candidate);
    retainedGhostKeys.add(key);
  }
}

const normalized = [];
for (const item of retained) {
  const rows = item.rows.map((row) => normalizeRow({ ...row }));
  if (item.entry.method_id === "csr5" || item.entry.method_id === "CSR5") {
    const dss = rows.find((row) => row.matrix_id === "Grund/d_ss");
    if (dss && dss.status !== "pass" && Number(dss.solve_ms) > 0) {
      dss.status = "pass";
      dss.solve_gflops = String(Number(dss.operations) / (Number(dss.solve_ms) * 1e6));
      dss.solve_only_efficiency_percent = String(Number(dss.solve_gflops) / Number(dss.peak_gflops) * 100);
    }
  }
  if (hasCoverage(rows)) {
    normalized.push({ entry: normalizeEntry(item.entry), headers: item.headers, rows });
  }
}

const cusparseCandidates = candidateLoaded
  .filter(({ entry }) => entry.candidate_group === "cusparse"
    && entry.selection_role !== "best")
  .map(({ entry, headers, rows }) => ({
    entry: normalizeEntry(entry),
    headers,
    rows: rows.map((row) => normalizeRow({ ...row })),
  }))
  .filter(({ rows }) => hasCoverage(rows));
const cusparseScopes = new Set(cusparseCandidates.map(({ entry }) =>
  `${operatorOf(entry)}:${entry.backend_id || canonicalBackend}:${datasetOf(entry)}`));
for (const scope of cusparseScopes) {
 const [operator, backend, dataset] = scope.split(":");
 for (const dtype of ["fp32", "fp64"]) {
  const candidates = cusparseCandidates.filter(({ entry }) =>
    (entry.backend_id || canonicalBackend) === backend
      && operatorOf(entry) === operator && datasetOf(entry) === dataset && entry.dtype === dtype);
  const availableConfigurations = new Set(
    candidates.map(({ entry }) => entry.configuration_id),
  );
  if (!candidates.length) continue;
  const selectedFrom = [...new Set(
    candidates.map(({ entry }) => entry.configuration_id).filter(Boolean),
  )].sort().join(",");
  const byMatrix = new Map();
  for (const candidate of candidates) {
    for (const row of candidate.rows.filter((value) =>
      value.status === "pass" && Number(value.solve_ms) > 0)) {
      const previous = byMatrix.get(row.matrix_id);
      if (!previous || Number(row.solve_ms) < Number(previous.row.solve_ms)) {
        byMatrix.set(row.matrix_id, {
          row: { ...row },
          configuration: candidate.entry.configuration_id,
        });
      }
    }
  }
  const rows = [...byMatrix.values()].map(({ row, configuration }) => {
    row.method_id = "cusparse-best";
    row.method_name = cuSparseName(backend, "BEST");
    row.configuration_id = "per-matrix-best";
    row.candidate_group = "cusparse";
    row.selection_role = "best";
    row.selected_from = configuration;
    row.source_kind = "derived";
    return normalizeRow(row);
  });
  rows.sort((a, b) => a.matrix_id.localeCompare(b.matrix_id));
  const source = candidates[0];
  if (source && rows.length) {
    const submissionId = "curated-cusparse-best-" + dtype;
    normalized.push({
      entry: {
        submission_id: submissionId,
        method_id: "cusparse-best",
        method_name: cuSparseName(backend, "BEST"),
        configuration_id: "per-matrix-best",
        candidate_group: "cusparse",
        selection_role: "best",
        selected_from: selectedFrom,
        base_format: "auto",
        operator_id: "spmv.csr." + dtype,
        dtype,
        backend_id: backend,
        hardware: source.entry.hardware || backend,
        peak_gflops: Number(source.entry.peak_gflops),
        dataset_id: source.entry.dataset_id,
        source_kind: "derived",
        source_manifest: source.entry.source_manifest || "",
        created_at: new Date().toISOString(),
      },
      headers: source.headers,
      rows: rows.map((row) => ({ ...row, submission_id: submissionId })),
    });
  }
 }
}

const ghostCandidates = ghostCandidateLoaded;
const ghostBestScopes = new Set(ghostCandidates.map(({ entry }) =>
  `${operatorOf(entry)}:${entry.backend_id || canonicalBackend}:${datasetOf(entry)}`));
for (const scope of ghostBestScopes) {
 const [operator, backend, dataset] = scope.split(":");
 for (const dtype of ["fp32", "fp64"]) {
  const candidates = ghostCandidates.filter(({ entry }) =>
    (entry.backend_id || canonicalBackend) === backend
      && operatorOf(entry) === operator && datasetOf(entry) === dataset && entry.dtype === dtype);
  const availableConfigurations = new Set(
    candidates.map(({ entry }) => entry.configuration_id),
  );
  if (!requiredGhostConfigurations.every((value) => availableConfigurations.has(value))) continue;
  const selectedFrom = [...new Set(
    candidates.map(({ entry }) => entry.configuration_id).filter(Boolean),
  )].sort().join(",");
  const byMatrix = new Map();
  for (const candidate of candidates) {
    for (const row of candidate.rows.filter((value) =>
      value.status === "pass" && Number(value.solve_ms) > 0)) {
      const previous = byMatrix.get(row.matrix_id);
      if (!previous || Number(row.solve_ms) < Number(previous.row.solve_ms)) {
        byMatrix.set(row.matrix_id, {
          row: { ...row },
          configuration: candidate.entry.configuration_id,
        });
      }
    }
  }
  const rows = [...byMatrix.values()].map(({ row, configuration }) => {
    row.method_id = "ghost-sell-c-sigma-best";
    row.method_name = "GHOST SELL-C-sigma BEST";
    row.configuration_id = "per-matrix-best";
    row.candidate_group = "ghost-sell";
    row.selection_role = "best";
    row.selected_from = configuration;
    row.base_format = "sell";
    row.source_kind = "derived";
    return normalizeRow(row);
  });
  rows.sort((a, b) => a.matrix_id.localeCompare(b.matrix_id));
  const source = candidates[0];
  if (source && rows.length) {
    const submissionId = "curated-ghost-sell-c-sigma-best-" + dtype;
    normalized.push({
      entry: {
        submission_id: submissionId,
        method_id: "ghost-sell-c-sigma-best",
        method_name: "GHOST SELL-C-sigma BEST",
        configuration_id: "per-matrix-best",
        candidate_group: "ghost-sell",
        selection_role: "best",
        selected_from: selectedFrom,
        base_format: "sell",
        operator_id: "spmv.csr." + dtype,
        dtype,
        backend_id: backend,
        hardware: source.entry.hardware || backend,
        peak_gflops: Number(source.entry.peak_gflops),
        dataset_id: source.entry.dataset_id,
        source_kind: "derived",
        source_manifest: source.entry.source_manifest || "",
        created_at: new Date().toISOString(),
      },
      headers: source.headers,
      rows: rows.map((row) => ({ ...row, submission_id: submissionId })),
    });
  }
 }
}

fs.rmSync(resultRoot, { recursive: true, force: true });
fs.mkdirSync(resultRoot, { recursive: true });
const submissions = [];
for (const item of normalized) {
  const entry = normalizeEntry(item.entry);
  const rows = item.rows.map((row) => ({
    ...row,
    submission_id: entry.submission_id,
  }));
  const backend = entry.backend_id || canonicalBackend;
  entry.backend_id = backend;
  const fileName = resultFileStem({ ...entry, backend_id: backend }) + ".csv";
  const relative = path.posix.join("data", "results", operatorOf(entry), backend, datasetOf(entry), fileName);
  const target = path.join(root, relative.replaceAll("/", path.sep));
  writeCsv(target, item.headers, rows);
  const first = rows[0];
  submissions.push({
    ...entry,
    path: relative,
    backend_id: backend,
    hardware: entry.hardware || backend,
    peak_gflops: Number(first.peak_gflops || item.entry.peak_gflops),
    created_at: item.entry.created_at || first.timestamp || new Date().toISOString(),
  });
}

manifest.generated_at = new Date().toISOString();
manifest.submissions = submissions;
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
console.log(JSON.stringify({
  backend: "multi-platform",
  submissions: submissions.length,
  selected_sell: [...selectedSell],
  csv_files: submissions.length,
}, null, 2));

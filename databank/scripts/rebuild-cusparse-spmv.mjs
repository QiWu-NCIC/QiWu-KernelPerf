import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const configurations = [
  "coo-alg1", "coo-alg2", "coo-default",
  "csc-alg1", "csc-alg2", "csc-default",
  "csr-alg1", "csr-alg2", "csr-default",
  ...[1, 128, 16, 2, 32, 4, 64, 8].flatMap((c) => [
    `sell-c${c}-alg1`, `sell-c${c}-default`,
  ]),
  "sell-nrows-alg1", "sell-nrows-default",
];

const dtypes = ["fp32", "fp64"];
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const publicRoot = path.join(root, "databank/public");
const kernelRoot = path.join(root, "KernelPerf");
const safe = (value) => String(value).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");

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
  return {
    headers,
    rows: records.map((values, index) => {
      if (values.length !== headers.length) throw new Error(`CSV width mismatch at row ${index + 2}`);
      return Object.fromEntries(headers.map((name, column) => [name, values[column]]));
    }),
  };
}

export function serializeCsv(headers, rows) {
  const escape = (value) => {
    const text = String(value ?? "");
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [headers.join(","), ...rows.map((row) => headers.map((header) => escape(row[header])).join(","))].join("\n") + "\n";
}

export function choosePerMatrixBest(rows) {
  const byMatrix = new Map();
  for (const row of rows) {
    if (!byMatrix.has(row.matrix_id)) byMatrix.set(row.matrix_id, []);
    byMatrix.get(row.matrix_id).push(row);
  }
  return [...byMatrix.entries()].map(([matrixId, candidates]) => {
    const passing = candidates.filter((row) => row.status === "pass" && Number(row.solve_ms) > 0);
    if (passing.length) {
      return passing.sort((left, right) => Number(left.solve_ms) - Number(right.solve_ms)
        || left.configuration_id.localeCompare(right.configuration_id))[0];
    }
    const failed = candidates.filter((row) => row.status === "fail" || row.status === "error");
    const preferred = failed.some((row) => row.status === "fail") ? "fail" : "error";
    return failed.filter((row) => row.status === preferred).sort((left, right) => (
      String(right.timestamp || "").localeCompare(String(left.timestamp || ""))
      || left.configuration_id.localeCompare(right.configuration_id)
    ))[0];
  }).filter(Boolean).sort((left, right) => left.matrix_id.localeCompare(right.matrix_id));
}

export function latestAttemptsPerMatrix(rows) {
  const byMatrix = new Map();
  for (const row of rows) {
    if (!byMatrix.has(row.matrix_id)) byMatrix.set(row.matrix_id, []);
    byMatrix.get(row.matrix_id).push(row);
  }
  return [...byMatrix.values()].map((attempts) => {
    const passing = attempts.filter((row) => row.status === "pass");
    const pool = passing.length ? passing : attempts.filter((row) => row.status === "fail" || row.status === "error");
    return pool.sort((left, right) => String(right.timestamp || "").localeCompare(String(left.timestamp || ""))
      || String(left.result_id || "").localeCompare(String(right.result_id || "")))[0];
  }).filter(Boolean).sort((left, right) => left.matrix_id.localeCompare(right.matrix_id));
}

export function chooseSellCWinner(rows) {
  const scoreByConfiguration = new Map();
  for (const row of rows) {
    const match = /^sell-c(1|2|4|8|16|32|64|128)-(alg1|default)$/.exec(row.configuration_id || "");
    if (!match || row.status !== "pass") continue;
    const efficiency = Number(row.solve_only_efficiency_percent);
    if (Number.isFinite(efficiency) && efficiency > 0) {
      const key = `${match[1]}|${match[2]}|${row.dtype}`;
      if (!scoreByConfiguration.has(key)) scoreByConfiguration.set(key, []);
      scoreByConfiguration.get(key).push(efficiency);
    }
  }
  const geometricMean = (values) => values.length
    ? Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length)
    : 0;
  const candidates = [];
  for (const c of [1, 2, 4, 8, 16, 32, 64, 128]) {
    const scores = ["alg1", "default"].flatMap((algorithm) => dtypes.map((dtype) => (
      geometricMean(scoreByConfiguration.get(`${c}|${algorithm}|${dtype}`) || [])
    )));
    if (scores.every((score) => score > 0)) candidates.push({ c, score: geometricMean(scores) });
  }
  candidates.sort((left, right) => right.score - left.score || left.c - right.c);
  if (!candidates.length) throw new Error("no SELL C has passing results for both algorithms and dtypes");
  return candidates[0].c;
}

function readIndex(indexPath) {
  return JSON.parse(fs.readFileSync(indexPath, "utf8"));
}

function writeIndex(indexPath, index) {
  const temporary = `${indexPath}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(index, null, 2) + "\n");
  try {
    fs.renameSync(temporary, indexPath);
  } catch (error) {
    if (process.platform !== "win32" || !["EPERM", "EEXIST"].includes(error.code)) throw error;
    fs.copyFileSync(temporary, indexPath);
    fs.unlinkSync(temporary);
  }
}

function indexEntry(row, relativePath, sourceEntry, overrides = {}) {
  return {
    submission_id: row.submission_id,
    path: relativePath,
    method_id: row.method_id,
    method_name: row.method_name,
    configuration_id: row.configuration_id,
    candidate_group: row.candidate_group,
    selection_role: row.selection_role,
    selected_from: row.selected_from || "",
    base_format: row.base_format,
    operator_id: row.operator_id,
    dtype: row.dtype,
    backend_id: row.backend_id,
    hardware: row.hardware,
    peak_gflops: Number(row.peak_gflops),
    dataset_id: row.dataset_id,
    source_kind: row.source_kind,
    created_at: row.timestamp || new Date().toISOString(),
    source_sha256: sourceEntry.source_sha256,
    source_manifest: sourceEntry.source_manifest,
    ...overrides,
  };
}

function writeScopedCsv(publicRootPath, relativePath, headers, rows) {
  const target = path.join(publicRootPath, relativePath.replaceAll("/", path.sep));
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, serializeCsv(headers, rows));
  return target;
}

export function rebuildCusparseCampaign({ inputRoot, backend, dataset, publicRootPath = publicRoot, kernelRootPath = kernelRoot }) {
  const datasetManifest = JSON.parse(fs.readFileSync(
    path.join(kernelRootPath, "config/datasets/suitesparse_all.json"), "utf8"
  ).replace(/^\uFEFF/, ""));
  if (dataset !== "suitesparse_all") throw new Error("only the official suitesparse_all corpus is supported");
  const expectedMatrices = new Set((Array.isArray(datasetManifest) ? datasetManifest : datasetManifest.matrices)
    .map((matrix) => matrix.matrix_id));
  const expectedCount = expectedMatrices.size;
  if (expectedCount !== 2904) throw new Error(`expected 2904 official matrices, found ${expectedCount}`);

  const files = [];
  const visit = (directory) => {
    for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
      const filename = path.join(directory, item.name);
      if (item.isDirectory()) visit(filename);
      else if (item.isFile() && item.name.toLowerCase().endsWith(".csv")) files.push(filename);
    }
  };
  visit(inputRoot);
  const headers = new Map();
  const groupedRows = new Map();
  for (const filename of files) {
    const parsed = parseCsv(fs.readFileSync(filename, "utf8"));
    for (const row of parsed.rows) {
      if (row.backend_id !== backend || row.dataset_id !== dataset || row.candidate_group !== "cusparse"
          || row.selection_role !== "candidate" || !configurations.includes(row.configuration_id)) continue;
      if (!dtypes.includes(row.dtype)) throw new Error(`unsupported dtype in ${filename}: ${row.dtype}`);
      const scope = `${row.configuration_id}|${row.dtype}`;
      const previousHeaders = headers.get(scope);
      if (previousHeaders && previousHeaders.join("|") !== parsed.headers.join("|")) {
        throw new Error(`CSV headers disagree for ${scope}`);
      }
      headers.set(scope, parsed.headers);
      if (!groupedRows.has(scope)) groupedRows.set(scope, []);
      groupedRows.get(scope).push(row);
    }
  }

  const candidates = new Map();
  for (const dtype of dtypes) {
    for (const configuration of configurations) {
      const scope = `${configuration}|${dtype}`;
      const attempts = groupedRows.get(scope) || [];
      if (!attempts.length) throw new Error(`missing candidate export: ${backend}/${dtype}/${configuration}`);
      const selected = latestAttemptsPerMatrix(attempts);
      const matrixIds = new Set(selected.map((row) => row.matrix_id));
      if (matrixIds.size !== expectedCount || [...expectedMatrices].some((id) => !matrixIds.has(id))) {
        throw new Error(`incomplete candidate ${backend}/${dtype}/${configuration}: ${matrixIds.size}/${expectedCount} matrices`);
      }
      const first = selected[0];
      if (selected.some((row) => row.method_id !== first.method_id || row.method_name !== first.method_name)) {
        throw new Error(`inconsistent method metadata for ${scope}`);
      }
      candidates.set(scope, { headers: headers.get(scope), rows: selected, first });
    }
  }

  const candidateIndexPath = path.join(publicRootPath, "data/candidate-pool/index.json");
  const publicIndexPath = path.join(publicRootPath, "data/index.json");
  const candidateIndex = readIndex(candidateIndexPath);
  const publicIndex = readIndex(publicIndexPath);
  const inScope = (entry) => entry.backend_id === backend && entry.dataset_id === dataset
    && entry.candidate_group === "cusparse" && entry.operator_id?.startsWith("spmv.");
  const sourceEntry = [...publicIndex.submissions, ...candidateIndex.submissions]
    .find((entry) => entry.backend_id === backend && entry.candidate_group === "cusparse"
      && entry.source_sha256 && entry.source_manifest);
  if (!sourceEntry) throw new Error(`no published cuSPARSE source package found for ${backend}`);

  const candidateEntries = [];
  const candidatePaths = new Set();
  for (const [scope, candidate] of candidates) {
    const [configuration, dtype] = scope.split("|");
    const rows = candidate.rows.map((row) => ({ ...row }));
    const first = rows[0];
    const relativePath = ["data/candidate-pool/spmv", backend, dataset,
      `${safe(first.method_id)}-${backend}-${dataset}-${dtype}.csv`].join("/");
    const submissionId = `full-${backend}-${dtype}-${configuration}`;
    rows.forEach((row) => { row.submission_id = submissionId; });
    const target = writeScopedCsv(publicRootPath, relativePath, candidate.headers, rows);
    const entry = indexEntry({ ...first, submission_id: submissionId }, relativePath, sourceEntry, {
      csv_sha256: crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex"),
      total: rows.length,
      passed: rows.filter((row) => row.status === "pass").length,
      failed: rows.filter((row) => row.status !== "pass").length,
    });
    candidateEntries.push(entry);
    candidatePaths.add(path.resolve(target));
  }

  const sellWinner = chooseSellCWinner([...candidates.values()].flatMap((candidate) => candidate.rows));
  const publicEntries = [];
  const publicPaths = new Set();
  const visibleConfigurations = configurations.filter((configuration) => (
    !configuration.startsWith("sell-c") || configuration.startsWith(`sell-c${sellWinner}-`)
  ));
  for (const configuration of visibleConfigurations) {
    for (const dtype of dtypes) {
      const candidate = candidates.get(`${configuration}|${dtype}`);
      const rows = candidate.rows.map((row) => ({ ...row }));
      const first = rows[0];
      const relativePath = ["data/results/spmv", backend, dataset,
        `${safe(first.method_id)}-${backend}-${dataset}-${dtype}.csv`].join("/");
      const submissionId = `full-${backend}-${dtype}-${configuration}`;
      rows.forEach((row) => { row.submission_id = submissionId; });
      const target = writeScopedCsv(publicRootPath, relativePath, candidate.headers, rows);
      publicEntries.push(indexEntry({ ...first, submission_id: submissionId }, relativePath, sourceEntry));
      publicPaths.add(path.resolve(target));
    }
  }

  const selectedFrom = configurations.join(",");
  const bestEntries = [];
  for (const dtype of dtypes) {
    const allRows = configurations.flatMap((configuration) => candidates.get(`${configuration}|${dtype}`).rows);
    const bestRows = choosePerMatrixBest(allRows).map((row) => ({
      ...row,
      submission_id: `cusparse-best-${backend}-${dataset}-${dtype}`,
      method_id: "cusparse-best",
      method_name: `cuSPARSE CUDA ${String(row.method_name).match(/CUDA\s+([^\s]+)/i)?.[1] || ""} BEST`.trim(),
      configuration_id: "per-matrix-best",
      candidate_group: "cusparse",
      selection_role: "best",
      selected_from: selectedFrom,
      base_format: "manual-selection",
      source_kind: "derived",
    }));
    const first = bestRows[0];
    const relativePath = ["data/results/spmv", backend, dataset,
      `cusparse-best-${backend}-${dataset}-${dtype}.csv`].join("/");
    const bestHeaders = candidates.get(`${configurations[0]}|${dtype}`).headers;
    const target = writeScopedCsv(publicRootPath, relativePath, bestHeaders, bestRows);
    const passed = bestRows.filter((row) => row.status === "pass").length;
    bestEntries.push(indexEntry(first, relativePath, sourceEntry, {
      method_id: "cusparse-best",
      method_name: first.method_name,
      configuration_id: "per-matrix-best",
      candidate_group: "cusparse",
      selection_role: "best",
      selected_from: selectedFrom,
      base_format: "manual-selection",
      source_kind: "derived",
      csv_sha256: crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex"),
      total: bestRows.length,
      passed,
      failed: bestRows.length - passed,
    }));
    publicPaths.add(path.resolve(target));
  }

  const oldPublicEntries = publicIndex.submissions.filter(inScope);
  const oldPublicPaths = oldPublicEntries.map((entry) => path.resolve(
    path.join(publicRootPath, entry.path.replaceAll("/", path.sep)),
  ));
  for (const filename of oldPublicPaths) {
    if (!publicPaths.has(filename) && fs.existsSync(filename)) fs.unlinkSync(filename);
  }
  candidateIndex.submissions = [
    ...candidateIndex.submissions.filter((entry) => !inScope(entry)),
    ...candidateEntries,
  ].sort((left, right) => left.path.localeCompare(right.path));
  publicIndex.submissions = [
    ...publicIndex.submissions.filter((entry) => !inScope(entry)),
    ...publicEntries,
    ...bestEntries,
  ].sort((left, right) => left.path.localeCompare(right.path));
  candidateIndex.generated_at = new Date().toISOString();
  publicIndex.generated_at = candidateIndex.generated_at;
  writeIndex(candidateIndexPath, candidateIndex);
  writeIndex(publicIndexPath, publicIndex);

  return {
    backend,
    dataset,
    matrices: expectedCount,
    configurations: configurations.length,
    dtype_scopes: candidates.size,
    sell_c_winner: sellWinner,
    public_candidates_per_dtype: visibleConfigurations.length,
    best_rows_per_dtype: Object.fromEntries(dtypes.map((dtype) => [
      dtype, bestEntries.find((entry) => entry.dtype === dtype)?.total || 0,
    ])),
  };
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index].startsWith("--")) args[argv[index].slice(2)] = argv[++index];
  }
  if (!args["input-root"] || !args.backend || !args.dataset) {
    throw new Error("usage: node scripts/rebuild-cusparse-spmv.mjs --input-root DIR --backend ID --dataset suitesparse_all");
  }
  return args;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = parseArgs(process.argv.slice(2));
    console.log(JSON.stringify(rebuildCusparseCampaign({
      inputRoot: path.resolve(args["input-root"]),
      backend: args.backend,
      dataset: args.dataset,
    }), null, 2));
  } catch (error) {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  }
}

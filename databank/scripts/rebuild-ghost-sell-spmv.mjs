import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { choosePerMatrixBest, latestAttemptsPerMatrix, parseCsv, serializeCsv } from "./rebuild-cusparse-spmv.mjs";

export const sigmaValues = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const dtypes = ["fp32", "fp64"];
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const publicRoot = path.join(root, "databank/public");
const expectedMatrices = new Set(JSON.parse(fs.readFileSync(
  path.join(root, "KernelPerf/config/datasets/suitesparse_all.json"), "utf8",
).replace(/^\uFEFF/, "")).map((matrix) => matrix.matrix_id));
const safe = (value) => String(value).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
const scopeKey = (sigma, dtype) => `${sigma}|${dtype}`;

export function isCompleteCoverage(rows, expected) {
  const matrices = new Set(rows.map((row) => row.matrix_id).filter(Boolean));
  return matrices.size === expected.size && [...expected].every((matrix) => matrices.has(matrix));
}

export function chooseOverallSigma(candidates) {
  const scores = [];
  for (const sigma of sigmaValues) {
    const dtypeScores = dtypes.map((dtype) => {
      const values = (candidates.get(scopeKey(sigma, dtype)) || [])
        .filter((row) => row.status === "pass")
        .map((row) => Number(row.solve_only_efficiency_percent))
        .filter((value) => Number.isFinite(value) && value > 0);
      return values.length
        ? Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length)
        : 0;
    });
    if (dtypeScores.some((score) => score <= 0)) continue;
    scores.push({
      sigma,
      score: Math.exp(dtypeScores.reduce((sum, score) => sum + Math.log(score), 0) / dtypeScores.length),
    });
  }
  scores.sort((left, right) => right.score - left.score || left.sigma - right.sigma);
  return scores[0] || null;
}

function readIndex(filename) {
  return JSON.parse(fs.readFileSync(filename, "utf8"));
}

function writeIndex(filename, index) {
  const temporary = `${filename}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(index, null, 2) + "\n");
  try {
    fs.renameSync(temporary, filename);
  } catch (error) {
    if (process.platform !== "win32" || !["EPERM", "EEXIST"].includes(error.code)) throw error;
    fs.copyFileSync(temporary, filename);
    fs.unlinkSync(temporary);
  }
}

function scanCsv(directory, visitor) {
  if (!fs.existsSync(directory)) return;
  for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
    const filename = path.join(directory, item.name);
    if (item.isDirectory()) scanCsv(filename, visitor);
    else if (item.isFile() && item.name.toLowerCase().endsWith(".csv")) visitor(filename);
  }
}

function collectRows({ inputRoots, backend, dataset, publicIndex, candidateIndex, publicRootPath }) {
  const grouped = new Map();
  const headers = new Map();
  const ingest = (filename) => {
    const parsed = parseCsv(fs.readFileSync(filename, "utf8"));
    for (const row of parsed.rows) {
      const match = /^sigma-(\d+)$/.exec(row.configuration_id || "");
      if (row.backend_id !== backend || row.dataset_id !== dataset
          || row.candidate_group !== "GHOST-SELL-C32-sigma" || row.operator_id?.split(".")[0] !== "spmv"
          || row.selection_role !== "candidate" || !match || !dtypes.includes(row.dtype)) continue;
      const sigma = Number(match[1]);
      if (!sigmaValues.includes(sigma)) continue;
      const key = scopeKey(sigma, row.dtype);
      const previousHeaders = headers.get(key);
      if (previousHeaders && previousHeaders.join("|") !== parsed.headers.join("|")) {
        throw new Error(`CSV headers disagree for sigma ${sigma}/${row.dtype}`);
      }
      headers.set(key, parsed.headers);
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    }
  };

  for (const directory of inputRoots) scanCsv(directory, ingest);
  const currentPublic = publicIndex.submissions.filter((entry) => entry.backend_id === backend
    && entry.dataset_id === dataset && entry.candidate_group === "GHOST-SELL-C32-sigma"
    && entry.operator_id?.startsWith("spmv."));
  const oldCandidatePool = candidateIndex.submissions.filter((entry) => entry.backend_id === backend
    && entry.dataset_id === dataset && entry.candidate_group === "GHOST-SELL-C32-sigma"
    && entry.operator_id?.startsWith("spmv."));
  for (const entry of [...currentPublic, ...oldCandidatePool]) {
    const filename = path.join(publicRootPath, entry.path.replaceAll("/", path.sep));
    if (fs.existsSync(filename)) ingest(filename);
  }

  const candidates = new Map();
  for (const [key, attempts] of grouped) {
    const selected = latestAttemptsPerMatrix(attempts);
    const ids = new Set(selected.map((row) => row.matrix_id));
    if (ids.size !== selected.length) throw new Error(`duplicate matrix after selection for ${key}`);
    candidates.set(key, selected);
  }
  return { candidates, headers, currentPublic, oldCandidatePool };
}

function entryFromRows(rows, relativePath, sourceEntry, publicRootPath, overrides = {}) {
  const first = rows[0];
  const target = path.join(publicRootPath, relativePath.replaceAll("/", path.sep));
  const passed = rows.filter((row) => row.status === "pass").length;
  return {
    submission_id: first.submission_id,
    path: relativePath,
    method_id: first.method_id,
    method_name: first.method_name,
    configuration_id: first.configuration_id,
    candidate_group: first.candidate_group,
    selection_role: first.selection_role,
    selected_from: first.selected_from || "",
    base_format: first.base_format,
    operator_id: first.operator_id,
    dtype: first.dtype,
    backend_id: first.backend_id,
    hardware: first.hardware,
    peak_gflops: Number(first.peak_gflops),
    dataset_id: first.dataset_id,
    source_kind: first.source_kind,
    created_at: first.timestamp || new Date().toISOString(),
    source_sha256: sourceEntry.source_sha256,
    source_manifest: sourceEntry.source_manifest,
    csv_sha256: crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex"),
    total: rows.length,
    passed,
    failed: rows.length - passed,
    public_ranked: first.public_ranked === "true",
    ...overrides,
  };
}

function writeRows(publicRootPath, relativePath, headers, rows) {
  const target = path.join(publicRootPath, relativePath.replaceAll("/", path.sep));
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, serializeCsv(headers, rows));
  return target;
}

export function rebuildGhostCampaign({ inputRoots, backend, dataset = "suitesparse_all", publicRootPath = publicRoot }) {
  if (dataset !== "suitesparse_all") throw new Error("only suitesparse_all is supported");
  const publicIndexPath = path.join(publicRootPath, "data/index.json");
  const candidateIndexPath = path.join(publicRootPath, "data/candidate-pool/index.json");
  const publicIndex = readIndex(publicIndexPath);
  const candidateIndex = readIndex(candidateIndexPath);
  const sourceEntry = [...publicIndex.submissions, ...candidateIndex.submissions].find((entry) => (
    entry.candidate_group === "GHOST-SELL-C32-sigma" && entry.source_sha256 && entry.source_manifest
  ));
  if (!sourceEntry) throw new Error("no GHOST source package is indexed");

  const { candidates, headers, currentPublic, oldCandidatePool } = collectRows({
    inputRoots,
    backend,
    dataset,
    publicIndex,
    candidateIndex,
    publicRootPath,
  });
  const candidateEntries = [];
  const candidatePaths = new Set();
  for (const [key, rows] of candidates) {
    const [sigmaText, dtype] = key.split("|");
    const sigma = Number(sigmaText);
    const first = rows[0];
    const submissionId = `full-${backend}-${dataset}-${dtype}-ghost-sigma-${sigma}`;
    const relativePath = ["data/candidate-pool/spmv", backend, dataset,
      `GHOST-SELL-C32-sigma-${sigma}-${backend}-${dataset}-${dtype}.csv`].join("/");
    const exportRows = rows.map((row) => ({ ...row, submission_id: submissionId }));
    const target = writeRows(publicRootPath, relativePath, headers.get(key), exportRows);
    candidateEntries.push(entryFromRows(exportRows, relativePath, sourceEntry, publicRootPath, {
      submission_id: submissionId,
      public_ranked: false,
      csv_sha256: crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex"),
    }));
    candidatePaths.add(path.resolve(target));
  }

  const complete = sigmaValues.every((sigma) => dtypes.every((dtype) => {
    const rows = candidates.get(scopeKey(sigma, dtype)) || [];
    return rows.length === expectedMatrices.size && isCompleteCoverage(rows, expectedMatrices);
  }));
  const generated = [];
  const generatedPaths = new Set();
  let overall = null;
  if (complete) {
    overall = chooseOverallSigma(candidates);
    if (overall) {
      const selectedFrom = sigmaValues.map((sigma) => `sigma-${sigma}`).join(",");
      for (const dtype of dtypes) {
        const selectedRows = choosePerMatrixBest(sigmaValues.flatMap((sigma) => candidates.get(scopeKey(sigma, dtype))));
        const outputs = [
          {
            id: "ghost-sell-c-sigma-best",
            name: "GHOST SELL-C-sigma BEST",
            configurationId: "per-matrix-best",
            metric: "per-matrix-solve-only",
            rows: selectedRows,
          },
          {
            id: "ghost-sell-c-sigma-overall-best",
            name: `GHOST SELL-C-sigma Overall BEST (sigma=${overall.sigma})`,
            configurationId: `overall-best-sigma-${overall.sigma}`,
            metric: "global-geomean-solve-efficiency",
            rows: candidates.get(scopeKey(overall.sigma, dtype)),
          },
        ];
        for (const output of outputs) {
          if (output.rows.length !== expectedMatrices.size || !isCompleteCoverage(output.rows, expectedMatrices)) {
            throw new Error(`incomplete derived result for ${backend}/${dtype}/${output.id}`);
          }
          const methodName = output.name;
          const submissionId = `full-${backend}-${dataset}-${dtype}-${output.id}`;
          const relativePath = ["data/results/spmv", backend, dataset,
            `${output.id}-${backend}-${dataset}-${dtype}.csv`].join("/");
          const rows = output.rows.map((row) => ({
            ...row,
            selected_configuration_id: row.configuration_id,
            submission_id: submissionId,
            method_id: output.id,
            method_name: methodName,
            configuration_id: output.configurationId,
            candidate_group: "ghost-sell",
            selection_role: "best",
            selection_metric: output.metric,
            selected_from: selectedFrom,
            base_format: "sell",
            source_kind: "derived",
            public_ranked: "true",
          }));
          const outputHeaders = [...headers.get(scopeKey(overall.sigma, dtype))];
          if (!outputHeaders.includes("selected_configuration_id")) outputHeaders.push("selected_configuration_id");
          const target = writeRows(publicRootPath, relativePath, outputHeaders, rows);
          const entry = entryFromRows(rows, relativePath, sourceEntry, publicRootPath, {
            submission_id: submissionId,
            method_id: output.id,
            method_name: methodName,
            configuration_id: output.configurationId,
            candidate_group: "ghost-sell",
            selection_role: "best",
            selected_from: selectedFrom,
            base_format: "sell",
            source_kind: "derived",
            public_ranked: true,
            csv_sha256: crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex"),
          });
          generated.push(entry);
          generatedPaths.add(path.resolve(target));
        }
      }
    }
  }

  const currentPaths = currentPublic.map((entry) => path.resolve(
    path.join(publicRootPath, entry.path.replaceAll("/", path.sep)),
  ));
  const oldDerived = publicIndex.submissions.filter((entry) => entry.backend_id === backend
    && entry.dataset_id === dataset && entry.candidate_group === "ghost-sell"
    && entry.operator_id?.startsWith("spmv."));
  for (const entry of oldDerived) currentPaths.push(path.resolve(
    path.join(publicRootPath, entry.path.replaceAll("/", path.sep)),
  ));
  const inScopePublic = (entry) => entry.backend_id === backend && entry.dataset_id === dataset
    && entry.operator_id?.startsWith("spmv.")
    && ["GHOST-SELL-C32-sigma", "ghost-sell"].includes(entry.candidate_group);
  publicIndex.submissions = [
    ...publicIndex.submissions.filter((entry) => !inScopePublic(entry)),
    ...generated,
  ].sort((left, right) => left.path.localeCompare(right.path));
  candidateIndex.submissions = [
    ...candidateIndex.submissions.filter((entry) => !oldCandidatePool.some((old) => (
      old.backend_id === entry.backend_id && old.dataset_id === entry.dataset_id
        && old.dtype === entry.dtype && old.configuration_id === entry.configuration_id
        && old.candidate_group === entry.candidate_group && old.operator_id === entry.operator_id
    ))),
    ...candidateEntries,
  ].sort((left, right) => left.path.localeCompare(right.path));

  const referencedPublic = new Set(publicIndex.submissions.map((entry) => path.resolve(
    path.join(publicRootPath, entry.path.replaceAll("/", path.sep)),
  )));
  for (const filename of currentPaths) {
    if (!referencedPublic.has(filename) && !candidatePaths.has(filename) && fs.existsSync(filename)) fs.unlinkSync(filename);
  }
  const generatedAt = new Date().toISOString();
  publicIndex.generated_at = generatedAt;
  candidateIndex.generated_at = generatedAt;
  writeIndex(publicIndexPath, publicIndex);
  writeIndex(candidateIndexPath, candidateIndex);

  return {
    backend,
    dataset,
    matrices_expected: expectedMatrices.size,
    sigma_count: sigmaValues.length,
    completed_scopes: [...candidates].filter(([key, rows]) => {
      const [sigma, dtype] = key.split("|");
      return sigmaValues.includes(Number(sigma)) && dtypes.includes(dtype)
        && rows.length === expectedMatrices.size && isCompleteCoverage(rows, expectedMatrices);
    }).length,
    candidate_scopes: candidateEntries.length,
    complete,
    overall_sigma: overall?.sigma ?? null,
    published_best_entries: generated.length,
  };
}

function parseArgs(argv) {
  const args = { inputRoots: [] };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--input-root") args.inputRoots.push(path.resolve(argv[++index]));
    else if (argv[index].startsWith("--")) args[argv[index].slice(2)] = argv[++index];
  }
  if (!args.inputRoots.length || !args.backend) {
    throw new Error("usage: node scripts/rebuild-ghost-sell-spmv.mjs --input-root DIR [--input-root DIR] --backend ID");
  }
  return args;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = parseArgs(process.argv.slice(2));
    console.log(JSON.stringify(rebuildGhostCampaign({
      inputRoots: args.inputRoots,
      backend: args.backend,
      dataset: args.dataset || "suitesparse_all",
    }), null, 2));
  } catch (error) {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  }
}

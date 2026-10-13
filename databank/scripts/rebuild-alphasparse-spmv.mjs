import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseCsv, serializeCsv } from "./rebuild-cusparse-spmv.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const publicRoot = path.join(root, "databank/public");
const backendIds = [
  "A100-SXM4-80GB", "H100-SXM5-80GB", "RTX5090-SL3061", "BW1000-gfx936", "Z100-gfx906",
];
const configurations = ["adaptive", "flat1", "flat4", "flat8", "line-enhance", "merge", "scalar", "vector"];

export function chooseBest(rows) {
  const byMatrix = new Map();
  for (const row of rows) {
    if (!byMatrix.has(row.matrix_id)) byMatrix.set(row.matrix_id, []);
    byMatrix.get(row.matrix_id).push(row);
  }
  return [...byMatrix.values()].map((candidates) => {
    const passing = candidates.filter((row) => row.status === "pass" && Number(row.solve_ms) > 0);
    if (passing.length) {
      passing.sort((left, right) => Number(left.solve_ms) - Number(right.solve_ms)
        || left.configuration_id.localeCompare(right.configuration_id));
      return passing[0];
    }
    const failed = candidates.filter((row) => row.status === "fail" || row.status === "error");
    failed.sort((left, right) => String(right.timestamp || "").localeCompare(String(left.timestamp || ""))
      || left.configuration_id.localeCompare(right.configuration_id));
    return failed[0];
  }).filter(Boolean).sort((left, right) => left.matrix_id.localeCompare(right.matrix_id));
}

export function rebuildAlphaSparse({ publicRootPath = publicRoot, backends = backendIds } = {}) {
  const expectedMatrices = new Set(JSON.parse(fs.readFileSync(
    path.join(root, "KernelPerf/config/datasets/suitesparse_all.json"), "utf8",
  ).replace(/^\uFEFF/, "")).map((matrix) => matrix.matrix_id));
  if (expectedMatrices.size !== 2904) throw new Error("unexpected suitesparse_all matrix manifest");
  const catalogPath = path.join(publicRootPath, "data/index.json");
  const index = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
  const sourceEntries = index.submissions.filter((entry) => entry.candidate_group === "AlphaSparseLib-CSR"
    && entry.selection_role === "candidate" && entry.dataset_id === "suitesparse_all");
  const generated = [];
  const generatedPaths = new Set();
  for (const backend of backends) for (const dtype of ["fp32", "fp64"]) {
    const candidates = sourceEntries.filter((entry) => entry.backend_id === backend && entry.dtype === dtype);
    const byConfiguration = new Map(candidates.map((entry) => [entry.configuration_id, entry]));
    if (configurations.some((configuration) => !byConfiguration.has(configuration))) continue;
    const rowsByConfiguration = new Map();
    let headers;
    for (const configuration of configurations) {
      const entry = byConfiguration.get(configuration);
      const parsed = parseCsv(fs.readFileSync(path.join(publicRootPath, entry.path), "utf8"));
      if (headers && headers.join("|") !== parsed.headers.join("|")) throw new Error("candidate CSV headers differ");
      headers = parsed.headers;
      const matrixIds = new Set(parsed.rows.map((row) => row.matrix_id));
      if (parsed.rows.length !== expectedMatrices.size || matrixIds.size !== expectedMatrices.size
          || [...expectedMatrices].some((matrix) => !matrixIds.has(matrix))) {
        throw new Error(`incomplete AlphaSparse candidate: ${backend}/${dtype}/${configuration}`);
      }
      rowsByConfiguration.set(configuration, parsed.rows);
    }
    const matrixSets = [...rowsByConfiguration.values()].map((rows) => new Set(rows.map((row) => row.matrix_id)));
    if (matrixSets.slice(1).some((set) => set.size !== matrixSets[0].size
        || [...matrixSets[0]].some((matrix) => !set.has(matrix)))) {
      throw new Error(`AlphaSparse matrix sets differ: ${backend}/${dtype}`);
    }
    const allRows = configurations.flatMap((configuration) => rowsByConfiguration.get(configuration));
    const bestRows = chooseBest(allRows).map((row) => ({
      ...row,
      submission_id: `alphasparse-best-${backend}-suitesparse_all-${dtype}`,
      method_id: "alphasparselib-csr-best",
      method_name: "AlphaSparseLib CSR BEST",
      configuration_id: "per-matrix-best",
      candidate_group: "AlphaSparseLib-CSR",
      selection_role: "best",
      selected_from: configurations.join(","),
      base_format: "manual-selection",
      source_kind: "derived",
    }));
    if (bestRows.length !== 2904) throw new Error(`incorrect AlphaSparse BEST coverage: ${backend}/${dtype}`);
    const relativePath = `data/results/spmv/${backend}/suitesparse_all/AlphaSparseLib-CSR-best-${backend}-suitesparse_all-${dtype}.csv`;
    const target = path.join(publicRootPath, relativePath);
    const text = serializeCsv(headers, bestRows);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, text);
    generatedPaths.add(relativePath);
    const source = byConfiguration.get(configurations[0]);
    generated.push({
      submission_id: bestRows[0].submission_id,
      path: relativePath,
      method_id: "alphasparselib-csr-best",
      method_name: "AlphaSparseLib CSR BEST",
      configuration_id: "per-matrix-best",
      candidate_group: "AlphaSparseLib-CSR",
      selection_role: "best",
      selected_from: configurations.join(","),
      base_format: "manual-selection",
      operator_id: `spmv.csr.${dtype}`,
      dtype,
      backend_id: backend,
      hardware: bestRows[0].hardware,
      peak_gflops: Number(bestRows[0].peak_gflops),
      dataset_id: "suitesparse_all",
      source_kind: "derived",
      source_sha256: source.source_sha256,
      source_manifest: source.source_manifest,
      created_at: new Date().toISOString(),
      total: bestRows.length,
      passed: bestRows.filter((row) => row.status === "pass").length,
      failed: bestRows.filter((row) => row.status !== "pass").length,
      csv_sha256: crypto.createHash("sha256").update(text).digest("hex"),
      public_ranked: true,
    });
  }
  const old = index.submissions.filter((entry) => entry.dataset_id === "suitesparse_all"
    && entry.candidate_group === "AlphaSparseLib-CSR" && entry.selection_role === "best");
  for (const entry of old) {
    if (generatedPaths.has(entry.path)) continue;
    const file = path.join(publicRootPath, entry.path);
    if (fs.existsSync(file)) fs.unlinkSync(file);
  }
  index.submissions = [...index.submissions.filter((entry) => !(entry.dataset_id === "suitesparse_all"
    && entry.candidate_group === "AlphaSparseLib-CSR" && entry.selection_role === "best")), ...generated]
    .sort((left, right) => left.path.localeCompare(right.path));
  index.generated_at = new Date().toISOString();
  fs.writeFileSync(catalogPath, JSON.stringify(index, null, 2) + "\n");
  return { generated: generated.length, backend_dtypes: generated.map((entry) => `${entry.backend_id}/${entry.dtype}`) };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { console.log(JSON.stringify(rebuildAlphaSparse(), null, 2)); }
  catch (error) { console.error(error.stack || error.message); process.exitCode = 1; }
}

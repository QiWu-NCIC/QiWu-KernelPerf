import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { choosePerMatrixBest, parseCsv, serializeCsv } from "./rebuild-cusparse-spmv.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const publicRoot = path.join(root, "databank/public");
const configurations = ["adaptive", "lrb", "stream"];
const dtypes = ["fp32", "fp64"];
const safe = (value) => String(value).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");

function writeIndex(filename, index) {
  fs.writeFileSync(filename, JSON.stringify(index, null, 2) + "\n");
}

export function rebuildRocSparse({ backend, dataset = "suitesparse_all", publicRootPath = publicRoot } = {}) {
  if (!backend) throw new Error("backend is required");
  if (dataset !== "suitesparse_all") throw new Error("only suitesparse_all is supported");
  const manifest = JSON.parse(fs.readFileSync(
    path.join(root, "KernelPerf/config/datasets/suitesparse_all.json"), "utf8",
  ).replace(/^\uFEFF/, ""));
  const expected = new Set(manifest.map((matrix) => matrix.matrix_id));
  if (expected.size !== 2904) throw new Error(`expected 2904 matrices, found ${expected.size}`);
  const indexPath = path.join(publicRootPath, "data/index.json");
  const index = JSON.parse(fs.readFileSync(indexPath, "utf8"));
  const candidates = new Map();
  for (const dtype of dtypes) for (const configuration of configurations) {
    const entry = index.submissions.find((item) => item.backend_id === backend
      && item.dataset_id === dataset && item.dtype === dtype
      && item.candidate_group === "rocSPARSE-DTK26.04-CSR"
      && item.configuration_id === configuration && item.selection_role === "candidate");
    if (!entry) throw new Error(`missing candidate entry: ${backend}/${dtype}/${configuration}`);
    const parsed = parseCsv(fs.readFileSync(path.join(publicRootPath, entry.path), "utf8"));
    const matrixIds = new Set(parsed.rows.map((row) => row.matrix_id));
    if (parsed.rows.length !== expected.size || matrixIds.size !== expected.size
      || [...expected].some((matrix) => !matrixIds.has(matrix))) {
      throw new Error(`incomplete candidate: ${backend}/${dtype}/${configuration} ${matrixIds.size}/2904`);
    }
    candidates.set(`${dtype}|${configuration}`, { entry, parsed });
  }
  const generated = [];
  const generatedPaths = new Set();
  for (const dtype of dtypes) {
    const first = candidates.get(`${dtype}|${configurations[0]}`);
    const rows = choosePerMatrixBest(configurations.flatMap((configuration) => (
      candidates.get(`${dtype}|${configuration}`).parsed.rows
    ))).map((row) => ({
      ...row,
      submission_id: `rocsparse-best-${backend}-${dataset}-${dtype}`,
      method_id: "rocSPARSE-DTK26.04-CSR-best",
      method_name: "rocSPARSE DTK 26.04 CSR BEST",
      configuration_id: "per-matrix-best",
      candidate_group: "rocSPARSE-DTK26.04-CSR",
      selection_role: "best",
      selected_from: configurations.join(","),
      base_format: "manual-selection",
      source_kind: "derived",
    }));
    if (rows.length !== expected.size) throw new Error(`incorrect BEST coverage: ${backend}/${dtype}`);
    const relativePath = ["data/results/spmv", backend, dataset,
      `rocSPARSE-DTK26.04-CSR-best-${backend}-${dataset}-${dtype}.csv`].join("/");
    const text = serializeCsv(first.parsed.headers, rows);
    const target = path.join(publicRootPath, relativePath);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, text);
    generatedPaths.add(path.resolve(target));
    const passed = rows.filter((row) => row.status === "pass").length;
    generated.push({
      ...first.entry,
      submission_id: rows[0].submission_id,
      path: relativePath,
      method_id: "rocSPARSE-DTK26.04-CSR-best",
      method_name: "rocSPARSE DTK 26.04 CSR BEST",
      configuration_id: "per-matrix-best",
      candidate_group: "rocSPARSE-DTK26.04-CSR",
      selection_role: "best",
      selected_from: configurations.join(","),
      base_format: "manual-selection",
      source_kind: "derived",
      total: rows.length,
      passed,
      failed: rows.length - passed,
      csv_sha256: crypto.createHash("sha256").update(text).digest("hex"),
      public_ranked: true,
    });
  }
  const inScope = (entry) => entry.backend_id === backend && entry.dataset_id === dataset
    && entry.candidate_group === "rocSPARSE-DTK26.04-CSR" && entry.selection_role === "best";
  for (const entry of index.submissions.filter(inScope)) {
    const filename = path.join(publicRootPath, entry.path);
    if (!generatedPaths.has(path.resolve(filename)) && fs.existsSync(filename)) fs.unlinkSync(filename);
  }
  index.submissions = [...index.submissions.filter((entry) => !inScope(entry)), ...generated]
    .sort((left, right) => left.path.localeCompare(right.path));
  index.generated_at = new Date().toISOString();
  writeIndex(indexPath, index);
  return { backend, dataset, generated: generated.length, rows_per_dtype: Object.fromEntries(
    generated.map((entry) => [entry.dtype, entry.total]),
  ) };
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index].startsWith("--")) args[argv[index].slice(2)] = argv[++index];
  }
  return args;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { console.log(JSON.stringify(rebuildRocSparse(parseArgs(process.argv.slice(2))), null, 2)); }
  catch (error) { console.error(error.stack || error.message); process.exitCode = 1; }
}

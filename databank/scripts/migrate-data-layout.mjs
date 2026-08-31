import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] || "public");
const dryRun = process.argv.includes("--dry-run");
const backends = new Map([
  ["ICT A100 GPU 0", "A100-SXM4-80GB"],
  ["ICT A100 SXM4 80GB", "A100-SXM4-80GB"],
  ["ict-a100", "A100-SXM4-80GB"],
  ["RTX 5090", "RTX5090-SL3061"],
  ["rtx5090", "RTX5090-SL3061"],
  ["H100 80GB", "H100-SXM5-80GB"],
  ["h100", "H100-SXM5-80GB"],
]);

const safe = (value) => String(value || "unknown").replace(/[^A-Za-z0-9._-]+/g, "-");
const canonicalBackend = (value) => {
  const text = String(value || "");
  return [...backends.entries()].find(([alias]) => alias.toLowerCase() === text.toLowerCase())?.[1]
    || text || "unknown-backend";
};
const operatorOf = (value) => String(value || "spmv").split(".")[0] || "spmv";
const datasetOf = (value) => safe(value || "unknown");
const fileStem = (entry) => [entry.method_id || entry.submission_id, entry.backend_id,
  entry.dataset_id, entry.dtype].map(safe).join("-");

function absolute(relative) {
  return path.join(root, relative.replaceAll("/", path.sep));
}

function relative(file) {
  return path.relative(root, file).replaceAll(path.sep, "/");
}

function move(source, target) {
  if (source === target) return false;
  if (!fs.existsSync(source)) throw new Error(`missing CSV: ${relative(source)}`);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  if (fs.existsSync(target)) {
    const sourceHash = crypto.createHash("sha256").update(fs.readFileSync(source)).digest("hex");
    const targetHash = crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex");
    if (sourceHash !== targetHash) {
      throw new Error(`target collision with different content: ${relative(target)}`);
    }
    if (!dryRun) fs.rmSync(source);
    return true;
  }
  if (!dryRun) fs.renameSync(source, target);
  return true;
}

function recoverCandidateSource(source, entry) {
  if (fs.existsSync(source)) return source;
  const directory = path.dirname(source);
  if (!fs.existsSync(directory)) return source;
  const matches = fs.readdirSync(directory)
    .filter((name) => name.endsWith(".csv"))
    .map((name) => path.join(directory, name))
    .filter((file) => {
      const header = fs.readFileSync(file, "utf8").split(/\r?\n/, 2).join("\n");
      return header.includes(`,${entry.method_id},`)
        && header.includes(`,${entry.dataset_id},`)
        && header.includes(`,${entry.dtype},`);
    });
  if (matches.length === 1) return matches[0];
  return source;
}

function migrateManifest(manifestRelative, kind) {
  const manifestFile = absolute(manifestRelative);
  const manifest = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
  const seenTargets = new Set();
  const plannedSources = new Set();
  let moved = 0;
  for (const entry of manifest.submissions || []) {
    const backend = canonicalBackend(entry.backend_id || entry.hardware);
    const dataset = datasetOf(entry.dataset_id);
    const operator = operatorOf(entry.operator_id);
    const stem = fileStem({ ...entry, backend_id: backend, dataset_id: dataset });
    const base = kind === "results"
      ? path.posix.join("data", "results", operator, backend, dataset)
      : path.posix.join("data", "candidate-pool", operator, backend, dataset);
    const targetRelative = path.posix.join(base, `${stem}.csv`);
    if (seenTargets.has(targetRelative)) throw new Error(`duplicate target: ${targetRelative}`);
    seenTargets.add(targetRelative);
    let source = absolute(entry.path);
    if (kind === "candidates") source = recoverCandidateSource(source, entry);
    plannedSources.add(path.resolve(source));
    const target = absolute(targetRelative);
    if (move(source, target)) moved += 1;
    entry.path = targetRelative;
  }
  if (!dryRun) {
    manifest.generated_at = new Date().toISOString();
    fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + "\n");
  }
  return {
    entries: (manifest.submissions || []).length,
    moved,
    manifest,
    plannedSources,
    plannedTargets: new Set([...seenTargets].map((value) => path.resolve(absolute(value)))),
  };
}

function listCsv(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((item) => {
    const file = path.join(directory, item.name);
    if (item.isDirectory()) return listCsv(file);
    return item.name.endsWith(".csv") ? [file] : [];
  });
}

const result = migrateManifest("data/index.json", "results");
const candidates = migrateManifest("data/candidate-pool/index.json", "candidates");
const candidateManifest = candidates.manifest;
const referenced = new Set([
  ...candidates.plannedSources,
  ...candidates.plannedTargets,
]);
const unreferenced = listCsv(absolute("data/candidate-pool"))
  .filter((file) => !referenced.has(path.resolve(file)));
for (const source of unreferenced) {
  // Unreferenced legacy files are intentionally removed: the published
  // catalog is the only supported source of downloadable benchmark data.
  if (!dryRun) fs.rmSync(source);
}

if (!dryRun) {
  const removeEmpty = (directory) => {
    if (!fs.existsSync(directory)) return;
    for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
      if (item.isDirectory()) removeEmpty(path.join(directory, item.name));
    }
    if (!fs.readdirSync(directory).length) fs.rmdirSync(directory);
  };
  removeEmpty(absolute("data/results"));
  removeEmpty(absolute("data/candidate-pool"));
}

console.log(JSON.stringify({
  dry_run: dryRun,
  results: { entries: result.entries, moved: result.moved },
  candidates: { entries: candidates.entries, moved: candidates.moved },
  removed_unreferenced_candidates: unreferenced.length,
}, null, 2));

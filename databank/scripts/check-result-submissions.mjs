import childProcess from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const inboxes = ["spmv", "spmm"].map((operator) => path.resolve("result-submissions", operator));
const requested = process.argv.slice(2);
const files = requested.length
  ? requested.map((file) => path.resolve(file))
  : inboxes.flatMap(walk).filter((file) => file.endsWith(".csv"));

const failures = [];
for (const file of files) {
  if (!file.endsWith(".csv") || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) {
    failures.push(`${file}: expected a readable .csv file`);
    continue;
  }
  const operator = inferOperator(file);
  if (!operator) {
    failures.push(`${file}: unable to infer SpMV or SpMM schema`);
    continue;
  }
  const result = childProcess.spawnSync(
    process.execPath,
    [path.resolve("scripts", `add-${operator}-result.mjs`), file, "--dry-run"],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    failures.push(`${path.relative(process.cwd(), file)}: ${(result.stderr || result.stdout).trim()}`);
    continue;
  }
  const summary = JSON.parse(result.stdout);
  console.log(`${path.relative(process.cwd(), file)} -> ${summary.target} (${summary.rows} rows)`);
}

console.log(JSON.stringify({ checked: files.length, failures }, null, 2));
if (failures.length) process.exitCode = 1;

function walk(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(file) : [file];
  });
}

function inferOperator(file) {
  const relative = path.relative(path.resolve("result-submissions"), file).replaceAll("\\", "/");
  if (relative.startsWith("spmv/")) return "spmv";
  if (relative.startsWith("spmm/")) return "spmm";
  const header = fs.readFileSync(file, "utf8").split(/\r?\n/, 1)[0];
  if (header.includes("rhs_columns") && header.includes("dense_layout")) return "spmm";
  if (header.includes("operator_id") && header.includes("effective_ms")) return "spmv";
  return "";
}

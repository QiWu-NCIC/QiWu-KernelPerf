import childProcess from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const inbox = path.resolve("result-submissions", "spmv");
const requested = process.argv.slice(2);
const files = requested.length
  ? requested.map((file) => path.resolve(file))
  : walk(inbox).filter((file) => file.endsWith(".csv"));

const failures = [];
for (const file of files) {
  if (!file.endsWith(".csv") || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) {
    failures.push(`${file}: expected a readable .csv file`);
    continue;
  }
  const result = childProcess.spawnSync(
    process.execPath,
    [path.resolve("scripts", "add-spmv-result.mjs"), file, "--dry-run"],
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

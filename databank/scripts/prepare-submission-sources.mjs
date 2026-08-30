import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repositoryRoot = path.resolve("..");
const submissionRoot = path.join(repositoryRoot, "KernelPerf", "submissions", "spmv");
const outputRoot = path.resolve("public", "source", "baselines");
const names = {
  alphasparse: "alphasparse",
  csr_adaptive: "csr-adaptive",
  csr5: "csr5",
  cusparse: "cusparse",
  ghost_sell: "ghost-sell",
};

fs.rmSync(path.dirname(outputRoot), { recursive: true, force: true });
for (const [submissionName, outputName] of Object.entries(names)) {
  const root = path.join(submissionRoot, submissionName);
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "submission.json"), "utf8"));
  const files = walk(root)
    .filter((file) => path.basename(file) !== "submission.json")
    .map((file) => ({ path: path.relative(root, file).replaceAll(path.sep, "/"), content: fs.readFileSync(file) }));
  const configurations = manifest.configurations_file
    ? JSON.parse(fs.readFileSync(path.join(root, manifest.configurations_file), "utf8"))
    : undefined;
  const digest = crypto.createHash("sha256");
  for (const file of files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0)) {
    digest.update(file.path).update("\0").update(file.content).update("\0");
  }
  const target = path.join(outputRoot, outputName);
  for (const file of files) {
    const destination = path.join(target, "files", ...file.path.split("/"));
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, file.content);
  }
  const plugin = {
    schema_version: 1,
    kind: "qiwu-spmv-source-plugin",
    snapshot_kind: "standalone-source-plugin",
    plugin_id: submissionName,
    operator_id: manifest.operator_id,
    supported_operators: manifest.supported_operators || [manifest.operator_id],
    base_format: manifest.base_format,
    entry_source: manifest.entry_source || "adapter.cu",
    compile_units: manifest.compile_units || [],
    build_profile: manifest.build_profile || "",
    candidate_group: manifest.candidate_group || "",
    configurations,
    source_sha256: digest.digest("hex"),
    source_path: `KernelPerf/submissions/spmv/${submissionName}`,
    files: files.map((file) => file.path),
  };
  fs.mkdirSync(target, { recursive: true });
  fs.writeFileSync(path.join(target, "plugin.json"), JSON.stringify(plugin, null, 2) + "\n");
}
console.log(JSON.stringify({ generated: Object.keys(names).length, output: outputRoot }, null, 2));

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((item) => {
    if ([".git", "build", "dist", "__pycache__"].includes(item.name)) return [];
    const file = path.join(directory, item.name);
    return item.isDirectory() ? walk(file) : [file];
  });
}

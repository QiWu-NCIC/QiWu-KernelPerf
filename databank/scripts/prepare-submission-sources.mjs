import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repositoryRoot = path.resolve("..");
const submissionRoot = path.join(repositoryRoot, "KernelPerf", "submissions", "spmv");
const outputRoot = path.resolve("public", "source", "baselines");
const textExtensions = new Set([
  ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inl", ".hip",
  ".json", ".md", ".py", ".sh", ".s", ".txt", ".in",
]);
const historicalSourceIds = {
  alphasparse: "0cf1fa4afffb1a77ed96445daf255408ca4037120bad2a95d2d43d69e71b33ca",
  cusparse: "66e0842cde8056240b7b0ac48c2aa766c1fa445b7935430db4a0380349e11d23",
};
const submissionNames = fs.readdirSync(submissionRoot, { withFileTypes: true })
  .filter((item) => item.isDirectory()
    && fs.existsSync(path.join(submissionRoot, item.name, "submission.json")))
  .map((item) => item.name)
  .sort();

// Baselines and content-addressed packages are regenerated from canonical
// submissions so a clean checkout satisfies both historical index formats.
fs.rmSync(outputRoot, { recursive: true, force: true });
for (const submissionName of submissionNames) {
  const outputName = submissionName.replaceAll("_", "-");
  const root = path.join(submissionRoot, submissionName);
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "submission.json"), "utf8"));
  const files = walk(root)
    .filter((file) => path.basename(file) !== "submission.json")
    .map((file) => {
      const relative = path.relative(root, file).replaceAll(path.sep, "/");
      return { path: relative, content: canonicalContent(relative, fs.readFileSync(file)) };
    });
  const configurations = manifest.configurations_file
    ? JSON.parse(fs.readFileSync(path.join(root, manifest.configurations_file), "utf8"))
    : undefined;
  const digest = crypto.createHash("sha256");
  for (const file of files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0)) {
    digest.update(file.path).update("\0").update(file.content).update("\0");
  }
  const sourceSha256 = digest.digest("hex");
  const target = path.join(outputRoot, outputName);
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
    include_dirs: manifest.include_dirs || [],
    build_profile: manifest.build_profile || "",
    candidate_group: manifest.candidate_group || "",
    upstream: manifest.upstream || null,
    configurations,
    source_sha256: sourceSha256,
    source_path: `KernelPerf/submissions/spmv/${submissionName}`,
    files: files.map((file) => file.path),
  };
  const targets = [target, path.resolve("public", "source", sourceSha256)];
  if (historicalSourceIds[submissionName]) {
    targets.push(path.resolve("public", "source", historicalSourceIds[submissionName]));
  }
  for (const targetRoot of targets) {
    for (const file of files) {
      const destination = path.join(targetRoot, "files", ...file.path.split("/"));
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      fs.writeFileSync(destination, file.content);
    }
    fs.mkdirSync(targetRoot, { recursive: true });
    fs.writeFileSync(path.join(targetRoot, "plugin.json"), JSON.stringify(plugin, null, 2) + "\n");
  }
}
console.log(JSON.stringify({ generated: submissionNames.length, output: outputRoot }, null, 2));

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((item) => {
    if ([".git", "build", "dist", "__pycache__"].includes(item.name)) return [];
    const file = path.join(directory, item.name);
    return item.isDirectory() ? walk(file) : [file];
  });
}

function canonicalContent(relative, content) {
  if (!textExtensions.has(path.extname(relative).toLowerCase())) return content;
  return Buffer.from(content.toString("utf8").replaceAll("\r\n", "\n").replaceAll("\r", "\n"));
}

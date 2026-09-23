import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repositoryRoot = path.resolve("..");
const submissionsRoot = path.join(repositoryRoot, "KernelPerf/submissions");
const spmmRoot = path.join(submissionsRoot, "spmm");
const outputRoot = path.resolve("public/source/spmm-baselines");
const catalogRoot = path.resolve("public/source/spmm");
fs.rmSync(outputRoot, { recursive: true, force: true });
fs.mkdirSync(catalogRoot, { recursive: true });
let generated = 0;
for (const item of fs.readdirSync(spmmRoot, { withFileTypes: true })) {
  if (!item.isDirectory()) continue;
  const root = path.join(spmmRoot, item.name);
  const manifestFile = path.join(root, "submission.json");
  if (!fs.existsSync(manifestFile)) continue;
  const manifest = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
  const configurations = manifest.configurations_file
    ? JSON.parse(fs.readFileSync(path.join(root, manifest.configurations_file), "utf8")) : [];
  for (const language of manifest.languages || [manifest.language || "cuda"]) {
    const files = new Map();
    const localPrefixes = manifest.source_include_by_language?.[language] || manifest.source_include;
    collect(root, localPrefixes, "", files);
    for (const shared of manifest.shared_source_roots || []) {
      const sharedRoot = path.resolve(submissionsRoot, shared.source);
      if (sharedRoot !== submissionsRoot && !sharedRoot.startsWith(submissionsRoot + path.sep)) throw new Error("shared source escapes submissions root");
      collect(sharedRoot, shared.include_by_language?.[language] || shared.include, shared.target || "", files);
    }
    const ordered = [...files].map(([file, content]) => ({ file, content })).sort((left, right) => left.file < right.file ? -1 : left.file > right.file ? 1 : 0);
    const digest = crypto.createHash("sha256");
    for (const file of ordered) digest.update(file.file).update("\0").update(file.content).update("\0");
    const sourceSha256 = digest.digest("hex");
    const compileUnits = manifest.compile_units_by_language?.[language] || manifest.compile_units || [];
    const includeDirs = manifest.include_dirs_by_language?.[language] || manifest.include_dirs || [];
    const entrySource = configurations[0]?.entry_source || manifest.entry_source || "adapter.cu";
    const plugin = {
      schema_version: 1, kind: "qiwu-spmm-source-plugin", snapshot_kind: "standalone-source-plugin",
      plugin_id: item.name + "-" + language, operator_id: manifest.operator_id,
      supported_operators: manifest.supported_operators || [manifest.operator_id],
      base_format: manifest.base_format, language, entry_source: entrySource,
      compile_units: compileUnits, include_dirs: includeDirs,
      build_profile: manifest.build_profile_by_language?.[language] || manifest.build_profile || "",
      candidate_group: manifest.candidate_group || "", upstream: manifest.upstream || null,
      configurations, source_sha256: sourceSha256,
      source_path: "KernelPerf/submissions/spmm/" + item.name,
      files: ordered.map((file) => file.file),
    };
    for (const target of [
      path.join(outputRoot, item.name.replaceAll("_", "-") + "-" + language),
      path.join(catalogRoot, sourceSha256),
    ]) {
      for (const file of ordered) {
        const destination = path.join(target, "files", ...file.file.split("/"));
        fs.mkdirSync(path.dirname(destination), { recursive: true });
        fs.writeFileSync(destination, file.content);
      }
      fs.mkdirSync(target, { recursive: true });
      fs.writeFileSync(path.join(target, "plugin.json"), JSON.stringify(plugin, null, 2) + "\n");
    }
    generated += 1;
  }
}
console.log(JSON.stringify({ operator: "spmm", generated, output: outputRoot }, null, 2));

function collect(root, prefixes, targetPrefix, files) {
  const normalizedPrefixes = Array.isArray(prefixes) ? prefixes.map((value) => value.replace(/\/$/, "")) : null;
  for (const file of walk(root)) {
    const relative = path.relative(root, file).replaceAll(path.sep, "/");
    if (["submission.json", "CMakeLists.txt", "README-QIWU-PLUGIN.md"].includes(relative)) continue;
    if (["examples/standalone.cu", "include/qiwu/spmv_plugin.cuh", "include/qiwu/spmm_plugin.cuh", "include/qiwu/gpu_runtime.h"].includes(relative)) continue;
    if (normalizedPrefixes && !normalizedPrefixes.some((prefix) => relative === prefix || relative.startsWith(prefix + "/"))) continue;
    const destination = [targetPrefix, relative].filter(Boolean).join("/");
    if (files.has(destination)) throw new Error("duplicate source path: " + destination);
    files.set(destination, fs.readFileSync(file));
  }
}
function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((item) => {
    if ([".git", "build", "dist", "node_modules", "__pycache__"].includes(item.name)) return [];
    const child = path.join(directory, item.name);
    if (item.isSymbolicLink()) throw new Error("source package contains symlink: " + child);
    return item.isDirectory() ? walk(child) : [child];
  });
}

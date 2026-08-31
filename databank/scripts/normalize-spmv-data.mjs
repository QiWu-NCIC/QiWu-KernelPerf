import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] || "public";
const canonical = {
  "ICT A100 GPU 0": "A100-SXM4-80GB",
  "ICT A100 SXM4 80GB": "A100-SXM4-80GB",
  "ict-a100": "A100-SXM4-80GB",
  "RTX 5090": "RTX5090-SL3061",
  "rtx5090": "RTX5090-SL3061",
  "H100 80GB": "H100-SXM5-80GB",
  "h100": "H100-SXM5-80GB",
};
const cudaVersion = (backend) => backend === "A100-SXM4-80GB" ? "12.9" : "12.8";

function normalizeEntry(entry) {
  const backend = canonical[entry.backend_id] || entry.backend_id;
  entry.backend_id = backend;
  if (entry.hardware) entry.hardware = canonical[entry.hardware] || backend;
  if (String(entry.method_name || "").startsWith("cuSPARSE ")) {
    entry.method_name = entry.method_name.replace(
      /^cuSPARSE(?: CUDA [0-9.]+)* /,
      `cuSPARSE CUDA ${cudaVersion(backend)} `,
    );
  }
  if (entry.path) {
    entry.path = entry.path.replace(/ICT[^/]*|RTX 5090|H100 80GB/g, backend);
  }
  return entry;
}

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((item) => {
    const file = path.join(directory, item.name);
    return item.isDirectory() ? walk(file) : [file];
  });
}

function normalizeCsv(file) {
  let text = fs.readFileSync(file, "utf8");
  const backend = Object.keys(canonical).find((value) => file.includes(value)) ||
    (file.includes("A100-SXM4-80GB") ? "A100-SXM4-80GB" : file.includes("H100") ? "H100-SXM5-80GB" : "RTX5090-SL3061");
  const version = cudaVersion(backend);
  text = text.replaceAll("ICT A100 GPU 0", "A100-SXM4-80GB")
    .replaceAll("ICT A100 SXM4 80GB", "A100-SXM4-80GB")
    .replaceAll("RTX 5090", "RTX5090-SL3061")
    .replaceAll("H100 80GB", "H100-SXM5-80GB")
    .replace(/,cuSPARSE(?: CUDA [0-9.]+)* /g, `,cuSPARSE CUDA ${version} `);
  fs.writeFileSync(file, text);
}

const indexFiles = [
  path.join(root, "data", "index.json"),
  path.join(root, "data", "candidate-pool", "index.json"),
];
for (const file of indexFiles) {
  if (!fs.existsSync(file)) continue;
  const document = JSON.parse(fs.readFileSync(file, "utf8"));
  if (Array.isArray(document.submissions)) document.submissions = document.submissions.map(normalizeEntry);
  if (Array.isArray(document.entries)) document.entries = document.entries.map(normalizeEntry);
  if (Array.isArray(document.candidates)) document.candidates = document.candidates.map(normalizeEntry);
  fs.writeFileSync(file, JSON.stringify(document, null, 2) + "\n");
}
for (const file of walk(path.join(root, "data"))) {
  if (file.endsWith(".csv")) normalizeCsv(file);
}
console.log(JSON.stringify({ normalized: true, root }, null, 2));

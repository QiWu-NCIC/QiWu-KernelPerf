import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const REQUIRED_COLUMNS = [
  "schema_version", "submission_id", "method_id", "method_name", "base_format",
  "operator_id", "dtype", "backend_id", "hardware", "peak_gflops", "job_id",
  "dataset_id", "matrix_id", "matrix_name", "rows", "cols", "nnz", "status", "operations",
  "preprocess_ms", "solve_ms", "solve_gflops", "solve_only_efficiency_percent",
  "timestamp", "source_kind",
];

const eventPath = process.env.ISSUE_EVENT_PATH || process.env.GITHUB_EVENT_PATH;
const issueNumber = process.env.ISSUE_NUMBER || "unknown";
const outputDir = process.env.OUTPUT_DIR || ".github/tmp";

if (!eventPath) {
  throw new Error("GITHUB_EVENT_PATH is required.");
}

const event = JSON.parse(fs.readFileSync(eventPath, "utf8"));
const body = event.issue?.body || "";
const csvPath = path.join(outputDir, `spmv-issue-${issueNumber}.csv`);

fs.mkdirSync(outputDir, { recursive: true });

const uploadSection = getIssueFormSection(body, "Result CSV upload");
const upload = findCsvLink(uploadSection) || findCsvLink(body);

if (upload) {
  const response = await fetchWithRetry(upload.url);

  if (!response.ok) {
    throw new Error(`Failed to download ${upload.url}: ${response.status} ${response.statusText}`);
  }

  const text = await response.text();
  validateCsvText(text, upload.label || upload.url);
  fs.writeFileSync(csvPath, text.endsWith("\n") ? text : `${text}\n`);
  writeGithubOutput("csv_path", csvPath);
  writeGithubOutput("csv_source", upload.url);
  console.log(`Extracted CSV attachment: ${upload.url}`);
} else {
  const pasted = cleanPastedCsv(getIssueFormSection(body, "Result CSV text"));
  validateCsvText(pasted, "Result CSV text");
  fs.writeFileSync(csvPath, pasted.endsWith("\n") ? pasted : `${pasted}\n`);
  writeGithubOutput("csv_path", csvPath);
  writeGithubOutput("csv_source", "issue body");
  console.log("Extracted CSV from issue body.");
}

function getIssueFormSection(markdown, label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`(?:^|\\n)### ${escaped}\\s*\\n([\\s\\S]*?)(?=\\n### |$)`, "i");
  return markdown.match(pattern)?.[1]?.trim() || "";
}

function findCsvLink(markdown) {
  const links = [];
  const markdownLinkPattern = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;
  let match;

  while ((match = markdownLinkPattern.exec(markdown))) {
    links.push({ label: match[1], url: match[2] });
  }

  const rawUrlPattern = /(https?:\/\/[^\s)]+(?:\.csv|\/[^\s)]*csv[^\s)]*))/g;
  while ((match = rawUrlPattern.exec(markdown))) {
    links.push({ label: match[1], url: match[1] });
  }

  return links.find((link) => /\.csv(?:[?#].*)?$/i.test(link.label) || /\.csv(?:[?#].*)?$/i.test(link.url));
}

function cleanPastedCsv(text) {
  const trimmed = text.trim();
  if (!trimmed || /^No response$/i.test(trimmed)) {
    return "";
  }

  const fenced = trimmed.match(/^```(?:csv)?\s*\n([\s\S]*?)\n```$/i);
  return (fenced?.[1] || trimmed).trim();
}

function validateCsvText(text, source) {
  const normalized = text.trim();
  if (!normalized) {
    throw new Error("No CSV attachment or pasted CSV content was found.");
  }

  const headers = normalized.split(/\r?\n/, 1)[0].trim().split(",");
  const missing = REQUIRED_COLUMNS.filter((column) => !headers.includes(column));
  if (missing.length) {
    throw new Error(
      source + " is missing required column(s): " + missing.join(", "),
    );
  }
}

async function fetchWithRetry(url, attempts = 3) {
  let lastError;
  const headers = process.env.GITHUB_TOKEN
    ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` }
    : {};

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, { headers });
      if (response.ok || attempt === attempts) {
        return response;
      }
      lastError = new Error(`${response.status} ${response.statusText}`);
    } catch (error) {
      lastError = error;
      if (attempt === attempts) {
        throw error;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, attempt * 2000));
  }

  throw lastError;
}

function writeGithubOutput(name, value) {
  const outputPath = process.env.GITHUB_OUTPUT;
  if (!outputPath) {
    return;
  }

  fs.appendFileSync(outputPath, `${name}=${value}\n`);
}

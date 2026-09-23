<template>
  <main class="spmm-page">
    <header class="page-header">
      <div>
        <p class="eyebrow">QiWu / Databank / SpMM</p>
        <h1>CSR SpMM benchmark</h1>
        <p>FP32/FP64 · CSR · row-major B/C<br />op(A)=N · op(B)=N</p>
      </div>
      <div class="actions">
        <button :disabled="busy || !index.submissions.length" @click="downloadArchive(index.submissions, 'qiwu-spmm-results.zip')">Download all SpMM results</button>
        <button :disabled="busy || !filteredEntries.length" @click="downloadArchive(filteredEntries, 'qiwu-spmm-filtered-' + filters.backend + '-n' + filters.rhs + '.zip')">Download filtered CSVs</button>
      </div>
    </header>

    <section class="filters" aria-label="SpMM filters">
      <label>Platform<select v-model="filters.backend"><option v-for="item in options.backend" :key="item">{{ item }}</option></select></label>
      <label>Dataset<select v-model="filters.dataset"><option v-for="item in options.dataset" :key="item">{{ item }}</option></select></label>
      <label>Dtype<select v-model="filters.dtype"><option v-for="item in options.dtype" :key="item">{{ item }}</option></select></label>
      <label>RHS columns N<select v-model.number="filters.rhs"><option v-for="item in options.rhs" :key="item" :value="item">{{ item }}</option></select></label>
      <label>Format<select v-model="filters.format"><option value="all">All</option><option v-for="item in options.format" :key="item">{{ item }}</option></select></label>
      <label>Base format<select v-model="filters.base"><option value="all">All</option><option v-for="item in options.base" :key="item">{{ item }}</option></select></label>
      <label>Timing<select v-model="filters.timing"><option value="solve-only">solve-only</option><option value="pre-plus-solve">pre+solve</option><option value="pre-amortized">pre/iteration+solve</option></select></label>
    </section>

    <section class="status">
      <span>{{ filteredEntries.length }} method CSVs</span>
      <span>{{ ranking.reduce((sum, item) => sum + item.passed, 0) }} passing matrices</span>
      <span>{{ ranking.reduce((sum, item) => sum + item.failed, 0) }} failed matrices</span>
      <span v-if="loading">Loading...</span><span v-if="error" class="error">{{ error }}</span>
    </section>

    <section class="protocol">
      <strong>Fixed protocol</strong>
      <span>N={{ filters.rhs }} - warmup 5 - measured 20 - seed 20260922 - FLOP=2*nnz*N</span>
      <span>CPU reference accumulates in long double; dynamic row bound C=4.</span>
      <span>N=1 is sanity-only and excluded from BEST.</span>
    </section>

    <section class="panel">
      <h2>Methods and BEST</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Method</th><th>Role</th><th>Pass / Fail</th><th>GFLOP/s</th><th>Efficiency</th><th>GPU</th><th>CPU</th><th>Library</th><th>Downloads</th></tr></thead>
          <tbody>
            <tr v-for="item in ranking" :key="item.entry.submission_id">
              <td><strong>{{ item.entry.method_name }}</strong><small>{{ item.entry.configuration_id || item.entry.selection_metric }}</small></td>
              <td>{{ item.entry.selection_role }}</td>
              <td>{{ item.passed }} / {{ item.failed }}</td>
              <td>{{ number(item.gflops) }}</td>
              <td>{{ number(item.efficiency) }}%</td>
              <td>{{ item.entry.hardware }}</td>
              <td>{{ item.entry.cpu_model || "unknown" }}</td>
              <td>{{ item.versions.join(", ") }}</td>
              <td class="download-cell"><a :href="asset(item.entry.path)" download>CSV</a><button @click="downloadSource(item.entry)">Source</button></td>
            </tr>
            <tr v-if="!ranking.length"><td colspan="9">No results match this exact SpMM scope.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Per-matrix results</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Matrix</th><th>Method</th><th>Status</th><th>nnz</th><th>Preprocess ms</th><th>{{ filters.timing }} ms</th><th>GFLOP/s</th><th>Validation</th><th>Selected config</th></tr></thead>
          <tbody>
            <tr v-for="row in visibleRows" :key="row.submission_id + row.matrix_id">
              <td>{{ row.matrix_name }}</td><td>{{ row.method_name }}</td><td :class="row.status === 'pass' ? 'pass' : 'fail'">{{ row.status }}</td>
              <td>{{ row.nnz }}</td><td>{{ number(row.preprocess_ms) }}</td><td>{{ number(row[timing.ms]) }}</td><td>{{ number(row[timing.gflops]) }}</td>
              <td>{{ row.validation_status }}<small v-if="row.error_type || row.failure_stage">{{ [row.error_type, row.failure_stage].filter(Boolean).join(" / ") }}</small><small v-if="row.error">{{ row.error }}</small></td><td>{{ row.selected_configuration_id || row.configuration_id }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
</template>

<script setup>
import JSZip from "jszip";
import { computed, onMounted, reactive, ref, watchEffect } from "vue";

const index = ref({ submissions: [] });
const rowsBySubmission = reactive(new Map());
const loading = ref(true);
const busy = ref(false);
const error = ref("");
const filters = reactive({ backend: "", dataset: "", dtype: "", rhs: 2, format: "all", base: "all", timing: "solve-only" });
const timings = {
  "solve-only": { ms: "solve_ms", gflops: "solve_gflops", efficiency: "solve_only_efficiency_percent" },
  "pre-plus-solve": { ms: "pre_plus_solve_ms", gflops: "pre_plus_solve_gflops", efficiency: "pre_plus_solve_efficiency_percent" },
  "pre-amortized": { ms: "pre_amortized_ms", gflops: "pre_amortized_gflops", efficiency: "pre_amortized_efficiency_percent" },
};
const timing = computed(() => timings[filters.timing]);
const unique = (values) => [...new Set(values.filter((value) => value !== undefined && value !== ""))].sort((left, right) => typeof left === "number" ? left - right : String(left).localeCompare(String(right)));
const options = computed(() => ({
  backend: unique(index.value.submissions.map((item) => item.backend_id)),
  dataset: unique(index.value.submissions.map((item) => item.dataset_id)),
  dtype: unique(index.value.submissions.map((item) => item.dtype)),
  rhs: unique(index.value.submissions.map((item) => Number(item.rhs_columns))),
  format: unique(index.value.submissions.map((item) => item.format)),
  base: unique(index.value.submissions.map((item) => item.base_format)),
}));
const filteredEntries = computed(() => index.value.submissions.filter((item) =>
  (!filters.backend || item.backend_id === filters.backend)
  && (!filters.dataset || item.dataset_id === filters.dataset)
  && (!filters.dtype || item.dtype === filters.dtype)
  && Number(item.rhs_columns) === Number(filters.rhs)
  && (filters.format === "all" || item.format === filters.format)
  && (filters.base === "all" || item.base_format === filters.base)));
const ranking = computed(() => filteredEntries.value.filter((entry) =>
  entry.public_ranked && entry.ranking_scope === "main").map((entry) => {
  const rows = rowsBySubmission.get(entry.submission_id) || [];
  const passedRows = rows.filter((row) => row.status === "pass");
  const mean = (field) => passedRows.length ? passedRows.reduce((sum, row) => sum + Number(row[field]), 0) / passedRows.length : 0;
  return { entry, passed: passedRows.length, failed: rows.length - passedRows.length,
    gflops: mean(timing.value.gflops), efficiency: mean(timing.value.efficiency),
    versions: unique(rows.map((row) => row.library_version)) };
}).sort((left, right) => right.efficiency - left.efficiency || left.entry.method_name.localeCompare(right.entry.method_name)));
const visibleRows = computed(() => filteredEntries.value.flatMap((entry) => rowsBySubmission.get(entry.submission_id) || [])
  .sort((left, right) => left.matrix_name.localeCompare(right.matrix_name) || left.method_name.localeCompare(right.method_name)));

function asset(relative) { return new URL(relative, new URL(import.meta.env.BASE_URL, window.location.origin)).toString(); }
function number(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "-"; }
function parseCsv(text) {
  const records = []; let record = [], cell = "", quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') { if (quoted && text[index + 1] === '"') { cell += '"'; index += 1; } else quoted = !quoted; }
    else if (character === "," && !quoted) { record.push(cell); cell = ""; }
    else if ((character === "\n" || character === "\r") && !quoted) { if (character === "\r" && text[index + 1] === "\n") index += 1; record.push(cell); if (record.some(Boolean)) records.push(record); record = []; cell = ""; }
    else cell += character;
  }
  if (record.length || cell) { record.push(cell); records.push(record); }
  const headers = records.shift() || [];
  return records.map((values) => Object.fromEntries(headers.map((name, column) => [name.replace(/^\uFEFF/, ""), values[column] || ""])));
}
function save(blob, filename) { const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = filename; link.click(); URL.revokeObjectURL(link.href); }
async function downloadArchive(entries, filename) {
  busy.value = true;
  try { const zip = new JSZip(); for (const entry of entries) zip.file(entry.submission_id + ".csv", await (await fetch(asset(entry.path))).text()); save(await zip.generateAsync({ type: "blob" }), filename); }
  catch (reason) { error.value = String(reason); } finally { busy.value = false; }
}
async function downloadSource(entry) {
  busy.value = true;
  try {
    const manifest = await (await fetch(asset(entry.source_manifest))).json();
    const root = entry.source_manifest.slice(0, entry.source_manifest.lastIndexOf("/"));
    const zip = new JSZip();
    zip.file("plugin.json", JSON.stringify(manifest, null, 2) + "\n");
    for (const file of manifest.files) zip.file(file, await (await fetch(asset(root + "/files/" + file))).arrayBuffer());
    save(await zip.generateAsync({ type: "blob" }), entry.method_id + "-source.zip");
  } catch (reason) { error.value = String(reason); } finally { busy.value = false; }
}
watchEffect(() => {
  for (const key of ["backend", "dataset", "dtype"]) if (options.value[key].length && !options.value[key].includes(filters[key])) filters[key] = options.value[key][0];
  if (options.value.rhs.length && !options.value.rhs.includes(Number(filters.rhs))) filters.rhs = options.value.rhs.find((value) => value !== 1) || options.value.rhs[0];
});
onMounted(async () => {
  try {
    index.value = await (await fetch(asset("data/spmm/index.json"))).json();
    await Promise.all(index.value.submissions.map(async (entry) => rowsBySubmission.set(entry.submission_id, parseCsv(await (await fetch(asset(entry.path))).text()))));
  } catch (reason) { error.value = String(reason); } finally { loading.value = false; }
});
</script>

<style scoped>
.spmm-page { box-sizing: border-box; width: 100%; min-height: 100vh; padding: 28px 5%; overflow-x: clip; color: #14213d; background: #f5f7fb; }
.page-header, .status, .protocol, .filters, .panel { box-sizing: border-box; max-width: 100%; min-width: 0; }
.page-header, .status, .protocol { display: flex; gap: 16px; align-items: center; justify-content: space-between; }
.page-header { align-items: flex-end; }
.page-header > div:first-child { flex: 1 1 auto; min-width: 0; }
.page-header p { width: 100%; max-width: 100%; white-space: normal; overflow-wrap: anywhere; }
.page-header h1 { margin: 4px 0; font-size: 32px; }
.eyebrow { color: #47679e; text-transform: uppercase; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.actions button, .download-cell button { padding: 7px 10px; border: 1px solid #98a2b3; border-radius: 4px; color: #14213d; background: white; font: inherit; font-size: 13px; cursor: pointer; }
.actions button:disabled { cursor: not-allowed; opacity: .55; }
.filters { display: grid; width: 100%; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px 16px; margin-top: 18px; padding: 16px 0; border-block: 1px solid #d5dbe5; }
.filters label { display: grid; min-width: 0; gap: 5px; font-size: 12px; font-weight: 700; }
.filters select { box-sizing: border-box; width: 100%; min-width: 0; padding: 8px; }
.status, .protocol { justify-content: flex-start; flex-wrap: wrap; margin: 14px 0; }
.status span, .protocol span, .protocol strong { min-width: 0; }
.protocol { padding: 10px 0; border-block: 1px solid #d5dbe5; }
.error, .fail { color: #b42318; }.pass { color: #067647; }
.panel { margin-top: 22px; padding-top: 16px; border-top: 1px solid #d5dbe5; }
.table-wrap { box-sizing: border-box; max-width: 100%; min-width: 0; overflow-x: auto; overscroll-behavior-x: contain; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; white-space: nowrap; }
th { position: sticky; top: 0; background: #eef3fc; }
td small { display: block; max-width: 340px; color: #667085; white-space: normal; }
.download-cell { display: flex; gap: 8px; }
@media (max-width: 760px) {
  .spmm-page { padding: 20px 4%; }
  .page-header { width: 100%; align-items: flex-start; flex-direction: column; }
  .page-header > div:first-child, .actions { box-sizing: border-box; width: 100%; flex: none; }
  .page-header h1 { font-size: 28px; }
  .actions { justify-content: flex-start; }
  .filters { grid-template-columns: minmax(0, 1fr); gap: 12px; }
  .status { align-items: flex-start; flex-direction: column; gap: 6px; }
  .protocol { align-items: flex-start; flex-direction: column; gap: 6px; }
  .panel { margin-top: 18px; padding-top: 14px; }
  .table-wrap table { min-width: 1000px; }
}
</style>

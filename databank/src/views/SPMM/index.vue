<template>
  <main class="spmv-page spmm-page">
    <header class="page-header">
      <div>
        <p class="eyebrow">QiWu / Databank</p>
        <h1>Contest</h1>
        <p class="header-copy">Normalized performance across submitted implementations and hardware.</p>
      </div>
      <nav class="header-actions" aria-label="SpMM actions">
        <a class="action-link" :href="submissionsUrl" target="_blank" rel="noopener noreferrer">Submit code</a>
        <a class="action-link" :href="resultSubmissionsUrl" target="_blank" rel="noopener noreferrer">Submit result</a>
        <button class="action-button" type="button" :disabled="downloading !== null" @click="downloadAllResults">
          {{ downloading ? "Preparing..." : "Download all SpMM CSVs" }}
        </button>
      </nav>
    </header>

    <OperatorTabs current="spmm" />

    <section class="control-band" aria-label="SpMM leaderboard filters">
      <label><span>Hardware</span><select v-model="filters.backend"><option v-for="value in backendOptions" :key="value" :value="value">{{ value }}</option></select></label>
      <label><span>Dataset</span><select v-model="filters.dataset"><option v-for="value in datasetOptions" :key="value" :value="value">{{ value }}</option></select></label>
      <label><span>RHS columns N</span><select v-model.number="filters.rhs"><option v-for="value in rhsOptions" :key="value" :value="value">{{ value }}</option></select></label>
      <label><span>Base format</span><select v-model="filters.baseFormat"><option value="all">All formats</option><option v-for="value in baseFormatOptions" :key="value" :value="value">{{ formatBaseFormat(value) }}</option></select></label>
      <div class="metric-note">
        <strong>Efficiency</strong>
        <span>FP32 + FP64 geomean</span>
        <small>SpMM FLOP = 2 &times; nnz &times; N</small>
      </div>
    </section>

    <section class="status-line" aria-live="polite">
      <span>{{ displayRows.length }} result rows</span>
      <span>{{ ranking.length }} methods with measured coverage</span>
      <span>{{ totalPassed }} passing cases</span>
      <span>{{ totalFailed }} failed cases</span>
      <span v-if="loading">Loading data...</span>
      <span v-if="error" class="error-text">{{ error }}</span>
    </section>

    <section class="protocol-section" aria-labelledby="protocol-title">
      <div class="protocol-heading">
        <div><p class="eyebrow">Reproducibility</p><h2 id="protocol-title">Benchmark protocol</h2></div>
        <span>SpMM FP32 / FP64</span>
      </div>
      <div class="protocol-grid">
        <div class="protocol-item">
          <span class="protocol-label">Timing &amp; sampling</span>
          <strong>{{ benchmarkProtocol.warmup }} warmups &middot; {{ benchmarkProtocol.iterations }} timed solves</strong>
          <p>CUDA Events (steady clock for host-timed plugins) measure solves; preprocessing uses host wall time plus stream synchronization.</p>
          <p>Warmup, validation and teardown are excluded.</p>
          <p>Row-major B/C; op(A)={{ benchmarkProtocol.opA }}, op(B)={{ benchmarkProtocol.opB }}, alpha={{ benchmarkProtocol.alpha }}, beta={{ benchmarkProtocol.beta }}. B seed {{ benchmarkProtocol.inputSeed }}.</p>
          <p>Preprocess, solve-only, and amortized preprocess-plus-solve are reported separately.</p>
        </div>
        <div class="protocol-item">
          <span class="protocol-label">Correctness</span>
          <strong>Independent CPU CSR reference</strong>
          <code>&rho;<sub>i,q</sub> = |C<sub>i,q</sub> &minus; C<sub>ref,i,q</sub>| / s<sub>i,q</sub> &le; {{ benchmarkProtocol.safetyFactor }} &middot; nnz<sub>i</sub> &middot; u</code>
          <p><em>ref</em> is stored as host <code>long double</code> for both dtypes; products are promoted before accumulation. Outputs use their declared FP32 or FP64 precision.</p>
          <p>If <em>s<sub>i,q</sub></em> = 0, <em>C<sub>i,q</sub></em> must be zero. NaN, Inf, runtime, dimension, and precision failures remain visible in source CSVs.</p>
        </div>
        <div class="protocol-item">
          <span class="protocol-label">Terms</span>
          <div class="protocol-definitions">
            <div class="protocol-definition"><code>C<sub>i,q</sub></code><span>GPU output element</span></div>
            <div class="protocol-definition"><code>C<sub>ref,i,q</sub></code><span>CPU reference</span></div>
            <div class="protocol-definition"><code>s<sub>i,q</sub> = &sum;<sub>j</sub>|a<sub>ij</sub>B<sub>j,q</sub>|</code><span>row scale</span></div>
            <div class="protocol-definition"><code>nnz<sub>i</sub></code><span>row nonzeros</span></div>
            <div class="protocol-definition"><code>u = &epsilon;(storage)/2</code><span>unit roundoff</span></div>
            <div class="protocol-definition"><code>C = {{ benchmarkProtocol.safetyFactor }}</code><span>safety factor</span></div>
            <div class="protocol-definition"><code>{{ benchmarkProtocol.flopsFormula }}</code><span>real-valued operation count</span></div>
            <div class="protocol-definition"><code>N=1</code><span>sanity only; excluded from ranking</span></div>
          </div>
        </div>
        <div class="protocol-item">
          <span class="protocol-label">Code locations</span>
          <div class="protocol-paths">
            <template v-for="source in protocolSources" :key="source.path + source.line">
              <a v-if="repositoryUrl" :href="sourceFileUrl(source)" target="_blank" rel="noopener noreferrer">{{ source.label }}</a>
              <code v-else>{{ source.label }}</code>
            </template>
          </div>
        </div>
      </div>
    </section>

    <section class="summary-section" aria-labelledby="leaderboard-title">
      <div class="section-heading summary-heading">
        <div><p class="eyebrow">Leading submissions</p><h2 id="leaderboard-title">Leaderboard</h2></div>
        <div class="summary-tools">
          <div class="ranking-mode" role="group" aria-label="Ranking time mode">
            <button v-for="mode in timeModes" :key="mode" type="button" :class="{ active: filters.timeMode === mode }" @click="chooseTimeMode(mode)">{{ mode === "pre/iteration+solve" ? `${mode} (${filters.iterations})` : mode }}</button>
          </div>
          <div class="pagination" aria-label="Leaderboard pages">
            <button type="button" aria-label="Previous page" :disabled="currentPage === 1" @click="currentPage -= 1">&larr;</button>
            <span>Page {{ currentPage }} / {{ pageCount }}</span>
            <button type="button" aria-label="Next page" :disabled="currentPage >= pageCount" @click="currentPage += 1">&rarr;</button>
          </div>
        </div>
      </div>
      <p v-if="ranking.length" class="table-summary">Showing {{ pageStart + 1 }}&ndash;{{ pageEnd }} of {{ ranking.length }} submissions</p>
      <div class="ranking-table-scroll">
        <table class="ranking-table">
          <thead><tr><th>Rank</th><th>Submission</th><th>Base format</th><th>Geo. mean GFLOP/s</th><th>Efficiency</th><th>Geo. mean preprocess</th><th>Geo. mean solve</th><th>Geo. mean effective</th><th>Passing</th></tr></thead>
          <tbody>
            <tr v-for="item in pagedRanking" :key="item.key" class="ranking-row" role="link" tabindex="0" :aria-label="`Show performance plots for ${item.methodName}`" @click="scrollToMethod(item.key)" @keydown.enter.prevent="scrollToMethod(item.key)" @keydown.space.prevent="scrollToMethod(item.key)">
              <td class="table-rank">{{ item.rank ?? "Unranked" }}</td>
              <td><strong>{{ item.methodName }}</strong><small v-if="item.configurationId">{{ item.configurationId }}<span v-if="item.selectionRole === 'best'"> &middot; per-matrix best</span></small></td>
              <td><span class="format-label">{{ formatBaseFormat(item.baseFormat) }}</span></td>
              <td>{{ formatMetricGflops(item.geomeanGflops) }}</td><td>{{ formatPercent(item.score) }}%</td>
              <td>{{ formatMilliseconds(item.geomeanPreprocessMs) }}</td><td>{{ formatMilliseconds(item.geomeanSolveMs) }}</td><td>{{ formatMilliseconds(item.geomeanEffectiveMs) }}</td>
              <td>{{ item.passCount }}/{{ item.expectedCaseCount }}<small v-if="item.failedCount">{{ item.failedCount }} failed</small><small v-if="item.missingCount">{{ item.missingCount }} missing results</small></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="!pagedRanking.length" class="empty-state">No published results match the selected filters.</p>
    </section>

    <section class="leaderboard-content">
      <div class="section-heading leaderboard-heading">
        <div>
          <p class="eyebrow">Performance distributions</p>
          <h2>Scatter ranking</h2>
        </div>
        <span class="score-unit">ranked by FP32 + FP64 efficiency geomean</span>
      </div>
      <div v-if="selectedPlotItem" class="method-bands">
        <article
          :id="methodBandId(selectedPlotItem.key)"
          :key="selectedPlotItem.key"
          class="method-band"
          tabindex="-1"
        >
          <header class="method-band-header">
            <div class="method-identity">
              <span class="rank-number">{{ selectedPlotItem.rank ?? "UR" }}</span>
              <div class="method-copy">
                <strong>{{ selectedPlotItem.methodName }}</strong>
                <span>GPU: {{ selectedPlotItem.hardware }} &middot; CPU: {{ selectedPlotItem.cpuModel || "not recorded" }} &middot; {{ formatBaseFormat(selectedPlotItem.baseFormat) }} &middot; {{ selectedPlotItem.dataset }} &middot; N={{ selectedPlotItem.rhsColumns }}</span>
                <small v-if="selectedPlotItem.configurationId">config: {{ selectedPlotItem.configurationId }}<span v-if="selectedPlotItem.selectionRole === 'best'"> &middot; per-matrix best</span></small>
                <small v-if="configurationDescription(selectedPlotItem)" class="config-provenance">{{ configurationDescription(selectedPlotItem) }}</small>
                <small v-if="selectedPlotItem.selectedFrom">selected from: {{ selectedPlotItem.selectedFrom }}</small>
                <small>{{ selectedPlotItem.passCount }}/{{ selectedPlotItem.expectedCaseCount }} passing cases &middot; {{ formatPercent(selectedPlotItem.coverageRatio * 100) }}% coverage<span v-if="selectedPlotItem.failedCount"> &middot; {{ selectedPlotItem.failedCount }} failed</span></small>
                <small v-if="selectedPlotItem.missingCount">{{ selectedPlotItem.missingCount }} missing results</small>
              </div>
            </div>
            <div class="method-band-actions">
              <div class="efficiency-score" :aria-label="selectedPlotItem.score > 0 ? `Efficiency ${formatPercent(selectedPlotItem.score)} percent` : 'No passing cases'">
                <span>Efficiency</span>
                <strong v-if="selectedPlotItem.score > 0">{{ formatPercent(selectedPlotItem.score) }}%</strong>
                <strong v-else>n/a</strong>
                <small v-if="selectedPlotItem.rankEligible">FP32 + FP64 geomean</small>
                <small v-else>Unranked: coverage below 90%</small>
              </div>
              <div class="download-actions">
                <button class="submission-download" type="button" :disabled="downloading !== null" @click="downloadSubmission(selectedPlotItem)">
                  {{ downloading === `result:${selectedPlotItem.key}` ? "Preparing..." : "Download result CSVs" }}
                </button>
                <button
                  v-if="selectedPlotItem.sourceManifests.length"
                  class="source-download"
                  type="button"
                  :disabled="downloading !== null || !selectedPlotItem.sourceManifests.length"
                  :title="selectedPlotItem.sourceManifests.length ? 'Download associated source package' : 'No source package was published for this result'"
                  @click="downloadSource(selectedPlotItem)"
                >{{ downloading === `source:${selectedPlotItem.key}` ? "Preparing..." : "Download source" }}</button>
              </div>
            </div>
          </header>
          <div class="method-plot-grid">
            <section v-for="dtype in dtypes" :key="dtype" class="dtype-plot">
              <div class="plot-card-heading">
                <h3>{{ dtype.toUpperCase() }}</h3>
                <span>{{ methodRows(selectedPlotItem, dtype).length }} cases &middot; GFLOP/s</span>
              </div>
              <canvas
                :ref="(element) => setCanvasRef(dtype, element)"
                class="scatter-canvas"
                :aria-label="`${selectedPlotItem.methodName} ${dtype.toUpperCase()} GFLOP/s scatter plot`"
              ></canvas>
            </section>
          </div>
        </article>
      </div>
      <p v-else class="empty-state">No published results match the selected filters.</p>
    </section>
    <button
      v-if="showBackToTop"
      class="back-to-top"
      type="button"
      aria-label="Back to top"
      title="Back to top"
      @click="scrollToTop"
    >
      &uarr;
    </button>
  </main>
</template>

<script setup>
import JSZip from "jszip";
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import OperatorTabs from "../../components/OperatorTabs.vue";
import { drawContestScatter } from "../../utils/contestCharts.js";
import { expectedMatrixCount } from "../../utils/contestCoverage.js";
import { selectContestEntries } from "../../utils/contestData.js";
import benchmarkSpecs from "../../../../KernelPerf/config/benchmarks.json";
import datasetSpecs from "../../../../KernelPerf/config/datasets.json";

const benchmark = benchmarkSpecs.find((item) => item.benchmark_id === "spmm");
const protocolsByDtype = Object.fromEntries((benchmark?.operators || []).map((operator) => [
  operator.dtype,
  { ...operator.metadata?.driver_config, flopsFormula: operator.flops_formula },
]));
const dtypes = ["fp32", "fp64"];
const timeModes = ["solve-only", "pre+solve", "pre/iteration+solve"];
const repositoryUrl = String(import.meta.env.VITE_REPOSITORY_URL || "https://github.com/QiWu-NCIC/QiWu-KernelPerf").replace(/\/$/, "");
const submissionsUrl = `${repositoryUrl}/tree/main/KernelPerf/submissions/spmm`;
const resultSubmissionsUrl = `${repositoryUrl}/upload/main/databank/result-submissions/spmm`;
const protocolSources = [
  { label: "KernelPerf/config/benchmarks.json:106", path: "KernelPerf/config/benchmarks.json", line: 106 },
  { label: "KernelPerf/benchmarks/spmm/template.cu:499", path: "KernelPerf/benchmarks/spmm/template.cu", line: 499 },
  { label: "KernelPerf/benchmarks/spmm/template.cu:518", path: "KernelPerf/benchmarks/spmm/template.cu", line: 518 },
  { label: "KernelPerf/benchmarks/spmm/template.cu:621", path: "KernelPerf/benchmarks/spmm/template.cu", line: 621 },
  { label: "KernelPerf/benchmarks/spmm/driver.py:654", path: "KernelPerf/benchmarks/spmm/driver.py", line: 654 },
  { label: "KernelPerf/benchmarks/spmm/driver.py:913", path: "KernelPerf/benchmarks/spmm/driver.py", line: 913 },
];
const rows = ref([]);
const submissionEntries = ref([]);
const platformMetadata = ref({});
const loading = ref(false);
const error = ref("");
const downloading = ref(null);
const showBackToTop = ref(false);
const filters = reactive({ backend: "", dataset: "suitesparse_sample_100", rhs: 2, baseFormat: "all", timeMode: "solve-only", iterations: 20 });
const currentPage = ref(1);
let resultController = null;
const pageSize = 20;
const canvasRefs = new Map();
const selectedMethodKey = ref("");
function sharedProtocolValue(field) {
  const values = dtypes.map((dtype) => protocolsByDtype[dtype]?.[field]);
  if (values.every((value) => value === values[0])) return values[0] ?? "not configured";
  return dtypes.map((dtype) => `${dtype.toUpperCase()} ${protocolsByDtype[dtype]?.[field] ?? "not configured"}`).join(" / ");
}

const benchmarkProtocol = {
  warmup: String(sharedProtocolValue("warmup")),
  iterations: String(sharedProtocolValue("iterations")),
  opA: sharedProtocolValue("op_a"),
  opB: sharedProtocolValue("op_b"),
  alpha: sharedProtocolValue("alpha"),
  beta: sharedProtocolValue("beta"),
  safetyFactor: String(sharedProtocolValue("validation_safety_factor")),
  inputSeed: sharedProtocolValue("input_seed"),
  flopsFormula: String(benchmark?.operators?.find((operator) => operator.dtype === "fp32")?.flops_formula || "2 * nnz * N").replaceAll("*", "x"),
};
function resultTimeMode() {
  if (filters.timeMode === "pre+solve") return "pre-plus-solve";
  if (filters.timeMode === "pre/iteration+solve") return "pre-amortized";
  return "solve-only";
}
const rhsOptions = computed(() => [...new Set(submissionEntries.value.map((entry) => Number(entry.rhs_columns)).filter((value) => Number.isFinite(value) && value > 1))].sort((left, right) => left - right));
const backendOptions = computed(() => [...new Set(submissionEntries.value.map((entry) => entry.backend_id).filter(Boolean))].sort());
const datasetOptions = computed(() => [...new Set(submissionEntries.value.map((entry) => entry.dataset_id).filter(Boolean))].sort());
const baseFormatOptions = computed(() => [...new Set(rows.value.map((row) => baseFormatKey(row.base_format)))].sort());
const downloadableEntries = computed(() => submissionEntries.value.filter((entry) =>
  Number(entry.rhs_columns) === Number(filters.rhs)
  && (entry.selection_role !== "best" || entry.selection_metric === resultTimeMode())
  && (!filters.backend || entry.backend_id === filters.backend)
  && (!filters.dataset || entry.dataset_id === filters.dataset)
  && (filters.baseFormat === "all" || baseFormatKey(entry.base_format) === filters.baseFormat)));
const scopedRows = computed(() => rows.value.filter((row) => Number(row.rhs_columns) === Number(filters.rhs)
  && (!filters.backend || row.backend_id === filters.backend)
  && (!filters.dataset || row.dataset_id === filters.dataset)));
const displayRows = computed(() => {
  const source = scopedRows.value.filter((row) => row.selection_role !== "best");
  const groups = new Map();
  for (const row of source) {
    if (row.selection_role !== "candidate" || row.ranking_scope !== "main" || !row.public_ranked) continue;
    const groupKey = [row.backend_id, row.dataset_id, row.dtype, row.rhs_columns, row.dense_layout, row.candidate_group].join("|");
    if (!groups.has(groupKey)) groups.set(groupKey, { candidates: [], winners: new Map(), configurations: new Set() });
    const group = groups.get(groupKey);
    group.candidates.push(row);
    group.configurations.add(row.configuration_id);
    if (row.status !== "pass" || effectiveMs(row) <= 0) continue;
    const previous = group.winners.get(row.matrix_id);
    if (!previous || effectiveMs(row) < effectiveMs(previous)
      || (effectiveMs(row) === effectiveMs(previous) && row.configuration_id < previous.configuration_id)) {
      group.winners.set(row.matrix_id, row);
    }
  }
  const bestRows = [];
  for (const [groupKey, group] of groups) {
    if (group.configurations.size < 2) continue;
    const [backendId, datasetId, dtype, rhsColumns, denseLayout, candidateGroup] = groupKey.split("|");
    const selectedFrom = [...group.configurations].sort().join(",");
    for (const row of group.winners.values()) {
      const methodId = `${candidateGroup}-BEST-${resultTimeMode()}`.replace(/[^A-Za-z0-9._-]+/g, "-");
      bestRows.push({
        ...row,
        submission_id: `derived-${[backendId, datasetId, dtype, rhsColumns, candidateGroup, resultTimeMode(), row.matrix_id].join("-")}`,
        method_id: methodId,
        method_name: presentationMethodName({ ...row, method_id: methodId, candidate_group: candidateGroup, configuration_id: "per-matrix-best" }),
        configuration_id: "per-matrix-best",
        candidate_group: candidateGroup,
        selection_role: "best",
        selection_metric: resultTimeMode(),
        selected_from: selectedFrom,
        selected_configuration_id: row.configuration_id,
        base_format: "manual-selection",
        backend_id: backendId,
        dataset_id: datasetId,
        dtype,
        rhs_columns: Number(rhsColumns),
        dense_layout: denseLayout,
        format: row.format,
        ranking_scope: "main",
        public_ranked: true,
        derived_best: true,
      });
    }
  }
  return [...source, ...bestRows].filter((row) => filters.baseFormat === "all"
    || baseFormatKey(row.base_format) === filters.baseFormat);
});
const filteredEntries = computed(() => submissionEntries.value.filter((entry) => Number(entry.rhs_columns) === Number(filters.rhs)
  && (entry.selection_role !== "best" || entry.selection_metric === resultTimeMode())
  && (!filters.backend || entry.backend_id === filters.backend)
  && (!filters.dataset || entry.dataset_id === filters.dataset)
  && (filters.baseFormat === "all" || baseFormatKey(entry.base_format) === filters.baseFormat)));
const totalPassed = computed(() => displayRows.value.filter((row) => row.status === "pass").length);
const totalFailed = computed(() => displayRows.value.length - totalPassed.value);

function setCanvasRef(dtype, element) {
  if (element) canvasRefs.set(dtype, element);
  else canvasRefs.delete(dtype);
}
function methodBandId(methodKey) {
  let hash = 2166136261;
  for (const character of String(methodKey)) { hash ^= character.codePointAt(0); hash = Math.imul(hash, 16777619); }
  return `method-band-${(hash >>> 0).toString(36)}`;
}
async function scrollToMethod(methodKey) {
  selectedMethodKey.value = methodKey;
  await nextTick();
  const target = document.getElementById(methodBandId(methodKey));
  if (target) { target.scrollIntoView({ behavior: "smooth", block: "start" }); target.focus({ preventScroll: true }); }
}
function updateBackToTopVisibility() { showBackToTop.value = window.scrollY > 320; }
function scrollToTop() { window.scrollTo({ top: 0, behavior: "smooth" }); }
function baseFormatKey(value) { return String(value || "unknown").trim().toLowerCase() || "unknown"; }
function formatBaseFormat(value) {
  const normalized = baseFormatKey(value);
  return normalized === "manual-selection" ? "Manual Selection" : normalized.toUpperCase();
}
function configurationLabel(configurationId) {
  const value = String(configurationId || "").replace(/^csr-?/i, "").replaceAll("-", " ").trim();
  return value ? value.toUpperCase() : "BEST";
}
function presentationMethodName(row) {
  const methodId = String(row.method_id || "");
  const candidateGroup = String(row.candidate_group || methodId);
  const configuration = configurationLabel(row.configuration_id);
  const sourceName = String(row.method_name || methodId);
  if (/alphasparse/i.test(candidateGroup)) return `AlphaSparseLib CSR ${configuration}`;
  if (/cusparse/i.test(candidateGroup)) {
    const version = sourceName.match(/CUDA\s+([\d.]+)/i)?.[1]
      || methodId.match(/CUDA[- ]([\d.]+)/i)?.[1]
      || String(row.library_version || "").match(/([\d]+(?:\.[\d]+)+)/)?.[1];
    return `cuSPARSE${version ? ` CUDA ${version}` : ""} CSR ${configuration}`.replace(/\s+/g, " ").trim();
  }
  if (/rocsparse/i.test(candidateGroup)) {
    const version = sourceName.match(/DTK\s+([\d.]+)/i)?.[1]
      || methodId.match(/DTK[- ]?([\d.]+)/i)?.[1];
    return `rocSPARSE${version ? ` DTK ${version}` : ""} CSR ${configuration}`.replace(/\s+/g, " ").trim();
  }
  return sourceName.replace(/\s+SpMM\s+/i, " ").replace(/\s+\[[^\]]+\]$/, "").replace(/\s+(?:solve-only|pre-plus-solve|pre-amortized)$/, "");
}
function configurationDescription(item) {
  const method = String(item.methodId || "").toLowerCase();
  if (method.includes("cusparse")) {
    const version = String(item.methodName || "").match(/CUDA\s+([\d.]+)/i)?.[1];
    return `NVIDIA cuSPARSE${version ? ` | CUDA ${version}` : ""} | https://docs.nvidia.com/cuda/cusparse/`;
  }
  if (method.includes("rocsparse")) {
    const version = String(item.methodName || "").match(/DTK\s+([\d.]+)/i)?.[1];
    return `ROCm rocSPARSE${version ? ` | DTK ${version}` : ""} | https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse`;
  }
  if (method.includes("alphasparse")) return "AlphaSparse/Library | https://github.com/AlphaSparse/Library | commit 248af573 (PR #31)";
  return "";
}
function formatPercent(value) { return Number(value || 0).toPrecision(5); }
function formatMetricGflops(value) {
  if (!value) return "0";
  if (value >= 100) return value.toFixed(1);
  if (value >= 1) return value.toFixed(3);
  if (value >= 0.001) return value.toFixed(5);
  return value.toExponential(3);
}
function formatMilliseconds(value) {
  if (!value) return "0 ms";
  if (value >= 100) return `${value.toFixed(1)} ms`;
  if (value >= 1) return `${value.toFixed(3)} ms`;
  if (value >= 0.001) return `${value.toFixed(5)} ms`;
  return `${value.toExponential(3)} ms`;
}
function assetUrl(relative) { return `${import.meta.env.BASE_URL}${relative}`; }
function sourceFileUrl(source) { return `${repositoryUrl}/blob/main/${source.path}#L${source.line}`; }
function parseCsv(text) {
  const records = []; let record = [], cell = "", quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') { if (quoted && text[index + 1] === '"') { cell += '"'; index += 1; } else quoted = !quoted; }
    else if (character === "," && !quoted) { record.push(cell); cell = ""; }
    else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      record.push(cell); if (record.some(Boolean)) records.push(record); record = []; cell = "";
    } else cell += character;
  }
  if (quoted) throw new Error("unterminated CSV quote");
  if (record.length || cell) { record.push(cell); records.push(record); }
  const headers = records.shift() || [];
  headers[0] = headers[0].replace(/^\uFEFF/, "");
  const parsed = records.map((values) => Object.fromEntries(headers.map((name, index) => [name, values[index] || ""])));
  Object.defineProperty(parsed, "headers", { value: headers });
  return parsed;
}
function normalizeRows(csv, entry) {
  const parsed = parseCsv(csv);
  return parsed.map((row) => ({
    ...row,
    csv_headers: parsed.headers,
    submission_id: row.submission_id || entry.submission_id,
    method_id: row.method_id || entry.method_id,
    method_name: presentationMethodName({ ...entry, ...row, method_name: row.method_name || entry.method_name }),
    configuration_id: row.configuration_id || entry.configuration_id,
    candidate_group: row.candidate_group || entry.candidate_group,
    selection_role: row.selection_role || entry.selection_role,
    selection_metric: row.selection_metric || entry.selection_metric,
    selected_from: row.selected_from || entry.selected_from,
    backend_id: row.backend_id || entry.backend_id,
    dataset_id: row.dataset_id || entry.dataset_id,
    dtype: row.dtype || entry.dtype,
    rhs_columns: Number(row.rhs_columns || entry.rhs_columns),
    format: row.format || entry.format,
    base_format: row.base_format || entry.base_format,
    hardware: row.hardware || entry.hardware,
    cpu_model: row.cpu_model || entry.cpu_model,
    peak_gflops: Number(row.peak_gflops || entry.peak_gflops || 0),
    operations: Number(row.operations || 0),
    nnz: Number(row.nnz || 0), rows: Number(row.rows || 0), cols: Number(row.cols || 0),
    preprocess_ms: Number(row.preprocess_ms || 0), solve_ms: Number(row.solve_ms || 0),
    pre_plus_solve_ms: Number(row.pre_plus_solve_ms || 0), pre_amortized_ms: Number(row.pre_amortized_ms || 0),
    solve_gflops: Number(row.solve_gflops || 0), pre_plus_solve_gflops: Number(row.pre_plus_solve_gflops || 0), pre_amortized_gflops: Number(row.pre_amortized_gflops || 0),
    solve_only_efficiency_percent: Number(row.solve_only_efficiency_percent || 0),
    pre_plus_solve_efficiency_percent: Number(row.pre_plus_solve_efficiency_percent || 0),
    pre_amortized_efficiency_percent: Number(row.pre_amortized_efficiency_percent || 0),
    public_ranked: row.public_ranked === "true" || entry.public_ranked === true,
    source_manifest: entry.source_manifest || "",
  }));
}
async function loadData() {
  loading.value = true; error.value = "";
  try {
    const base = import.meta.env.BASE_URL;
    const [manifestResponse, platformsResponse] = await Promise.all([
      fetch(`${base}data/results/spmm/index.json?v=${Date.now()}`, { cache: "no-store" }),
      fetch(`${base}data/platforms.json?v=${Date.now()}`, { cache: "no-store" }),
    ]);
    if (!manifestResponse.ok) throw new Error(`SpMM manifest: ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();
    if (platformsResponse.ok) {
      const platforms = await platformsResponse.json();
      platformMetadata.value = Object.fromEntries((platforms.platforms || []).map((item) => [item.backend_id, item]));
    }
    const latest = new Map();
    for (const entry of manifest.submissions || []) {
      const key = [entry.method_id, entry.configuration_id, entry.backend_id, entry.dataset_id, entry.dtype,
        entry.rhs_columns, entry.dense_layout, entry.selection_metric].join("|");
      const previous = latest.get(key);
      const quality = (item) => [Number(item.passed || 0), Number(item.total || 0), String(item.created_at || "")];
      const currentQuality = quality(entry);
      const previousQuality = previous ? quality(previous) : [];
      if (!previous || currentQuality.some((value, index) => value > previousQuality[index]
        && currentQuality.slice(0, index).every((part, previousIndex) => part === previousQuality[previousIndex]))) {
        latest.set(key, entry);
      }
    }
    submissionEntries.value = [...latest.values()];
    if (!backendOptions.value.includes(filters.backend)) filters.backend = backendOptions.value[0] || "";
    if (!datasetOptions.value.includes(filters.dataset)) filters.dataset = datasetOptions.value[0] || "";
    if (!rhsOptions.value.includes(Number(filters.rhs))) filters.rhs = rhsOptions.value[0] ?? 2;
  } catch (cause) { error.value = `Unable to load SpMM result data: ${cause.message}`; }
  finally { if (!resultController) loading.value = false; await nextTick(); drawSelected(); }
}
async function loadSelectedData() {
  resultController?.abort();
  const controller = new AbortController();
  resultController = controller;
  loading.value = true;
  error.value = "";
  rows.value = [];
  try {
    const entries = selectContestEntries(submissionEntries.value, filters)
      .filter((entry) => entry.selection_role !== "best");
    const loaded = await Promise.allSettled(entries.map(async (entry) => {
      const response = await fetch(`${assetUrl(entry.path)}?v=${encodeURIComponent(entry.csv_sha256 || entry.submission_id)}`,
        { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`${entry.path}: ${response.status}`);
      return normalizeRows(await response.text(), entry);
    }));
    if (controller.signal.aborted) return;
    rows.value = loaded.filter((result) => result.status === "fulfilled").flatMap((result) => result.value);
    const failures = loaded.filter((result) => result.status === "rejected");
    if (failures.length) error.value = `${failures.length} result file(s) could not be loaded.`;
  } finally {
    if (!controller.signal.aborted) loading.value = false;
  }
}
function metric(row, field) { return Number(row[field] || 0); }
function effectiveMs(row) {
  if (filters.timeMode === "pre+solve") return row.preprocess_ms + row.solve_ms;
  if (filters.timeMode === "pre/iteration+solve") return row.preprocess_ms / filters.iterations + row.solve_ms;
  return row.solve_ms;
}
function chooseTimeMode(mode) {
  if (mode === "pre/iteration+solve") {
    const value = Number.parseInt(window.prompt("Preprocess amortization iteration count", String(filters.iterations)), 10);
    if (!Number.isFinite(value) || value <= 0) return;
    filters.iterations = value;
  }
  filters.timeMode = mode;
}
function performanceGflops(row) {
  const time = effectiveMs(row);
  return row.status === "pass" && time > 0 && row.operations > 0 ? row.operations / (time * 1e6) : 0;
}
function efficiency(row) {
  const performance = performanceGflops(row);
  return performance > 0 && row.peak_gflops > 0 ? performance / row.peak_gflops * 100 : 0;
}
function geometricMean(values) { const valid = values.filter((value) => value > 0); return valid.length ? Math.exp(valid.reduce((sum, value) => sum + Math.log(value), 0) / valid.length) : 0; }
function methodRows(item, dtype) { return displayRows.value.filter((row) => row.dtype === dtype
  && row.method_id === item.methodId && row.configuration_id === item.configurationId
  && row.backend_id === item.backendId && row.dataset_id === item.dataset
  && Number(row.rhs_columns) === item.rhsColumns && row.format === item.format
  && baseFormatKey(row.base_format) === baseFormatKey(item.baseFormat)
  && performanceGflops(row) > 0); }
const ranking = computed(() => {
  const groups = new Map();
  const expected = new Map();
  for (const row of scopedRows.value) {
    if (row.selection_role !== "candidate" || row.ranking_scope !== "main" || !row.public_ranked) continue;
    const scope = [row.backend_id, row.dataset_id, row.rhs_columns, row.dtype].join("|");
    if (!expected.has(scope)) expected.set(scope, new Set());
    expected.get(scope).add(row.matrix_id);
  }
  for (const row of displayRows.value) {
    if (row.ranking_scope !== "main" || !row.public_ranked) continue;
    const configurationId = row.configuration_id || "";
    const methodKey = [row.method_id, configurationId, row.backend_id, row.dataset_id,
      row.rhs_columns, baseFormatKey(row.base_format), row.selection_metric || ""].join("|");
    if (!groups.has(methodKey)) groups.set(methodKey, {
      key: methodKey, methodId: row.method_id, configurationId,
      methodName: row.method_name || row.method_id, backendId: row.backend_id, hardware: row.hardware,
      cpuModel: row.cpu_model || platformMetadata.value[row.backend_id]?.cpu_model || "",
      dataset: row.dataset_id, rhsColumns: Number(row.rhs_columns), format: row.format,
      baseFormat: row.base_format, candidateGroup: row.candidate_group || "",
      selectionRole: row.selection_role || "candidate",
      selectionMetric: row.selection_metric || "", selectedFrom: row.selected_from || "",
      byDtype: new Map(), matricesByDtype: new Map(), performanceValues: [], preprocessValues: [], solveValues: [], effectiveValues: [],
      sourceManifests: new Set(), submissionIds: new Set(), resultRows: [], libraryVersions: new Set(),
    });
    const item = groups.get(methodKey);
    item.resultRows.push(row);
    if (!item.byDtype.has(row.dtype)) item.byDtype.set(row.dtype, []);
    if (!item.matricesByDtype.has(row.dtype)) item.matricesByDtype.set(row.dtype, new Set());
    if (row.status === "pass" && performanceGflops(row) > 0) {
      item.byDtype.get(row.dtype).push(efficiency(row));
      item.matricesByDtype.get(row.dtype).add(row.matrix_id);
      item.performanceValues.push(performanceGflops(row));
      item.preprocessValues.push(row.preprocess_ms);
      item.solveValues.push(row.solve_ms);
      item.effectiveValues.push(effectiveMs(row));
    }
    if (row.source_manifest) item.sourceManifests.add(row.source_manifest);
    if (row.submission_id && !String(row.submission_id).startsWith("derived-")) item.submissionIds.add(row.submission_id);
    if (row.library_version) item.libraryVersions.add(row.library_version);
  }
  return [...groups.values()].map((item) => {
    const scores = dtypes.map((dtype) => geometricMean(item.byDtype.get(dtype) || [])).filter((score) => score > 0);
    const coverageByDtype = dtypes.map((dtype) => {
      const expectedCount = expectedMatrixCount(item.dataset,
        expected.get([item.backendId, item.dataset, item.rhsColumns, dtype].join("|"))?.size || 0, datasetSpecs);
      return expectedCount ? (item.matricesByDtype.get(dtype)?.size || 0) / expectedCount : 0;
    });
    const expectedCaseCount = dtypes.reduce((sum, dtype) => sum
      + expectedMatrixCount(item.dataset,
        expected.get([item.backendId, item.dataset, item.rhsColumns, dtype].join("|"))?.size || 0, datasetSpecs), 0);
    const passCount = [...item.matricesByDtype.values()].reduce((sum, matrices) => sum + matrices.size, 0);
    const failedCount = item.resultRows.filter((row) => row.status !== "pass").length;
    const missingCount = Math.max(0, expectedCaseCount - item.resultRows.length);
    const coverageRatio = expectedCaseCount ? passCount / expectedCaseCount : 0;
    const rankEligible = coverageByDtype.every((coverage) => coverage >= 0.9) && scores.length === dtypes.length;
    return {
      ...item,
      sourceManifests: [...item.sourceManifests],
      submissionIds: [...item.submissionIds],
      libraryVersions: [...item.libraryVersions].sort(),
      expectedCaseCount,
      passCount,
      failedCount,
      missingCount,
      coverageRatio,
      coverageByDtype,
      rankEligible,
      geomeanGflops: geometricMean(item.performanceValues),
      geomeanPreprocessMs: geometricMean(item.preprocessValues),
      geomeanSolveMs: geometricMean(item.solveValues),
      geomeanEffectiveMs: geometricMean(item.effectiveValues),
      score: geometricMean(scores),
    };
  }).filter((item) => item.passCount > 0 || item.failedCount > 0)
    .sort((left, right) => Number(right.rankEligible) - Number(left.rankEligible)
      || right.score - left.score || right.passCount - left.passCount)
    .reduce((items, item) => {
      const rankedCount = items.filter((value) => value.rank !== null).length;
      items.push({ ...item, rank: item.rankEligible ? rankedCount + 1 : null });
      return items;
    }, []);
});
const pageCount = computed(() => Math.max(1, Math.ceil(ranking.value.length / pageSize)));
const pageStart = computed(() => (currentPage.value - 1) * pageSize);
const pageEnd = computed(() => Math.min(pageStart.value + pageSize, ranking.value.length));
const pagedRanking = computed(() => ranking.value.slice(pageStart.value, pageEnd.value));
const selectedPlotItem = computed(() => ranking.value.find((item) => item.key === selectedMethodKey.value)
  || ranking.value[0]
  || null);
function drawSelected() {
  const item = selectedPlotItem.value;
  if (!item) return;
  for (const dtype of dtypes) {
    drawContestScatter(canvasRefs.get(dtype), methodRows(item, dtype), performanceGflops);
  }
}
function safeFileComponent(value) { return String(value || "result").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "result"; }
function resultFileName(entry) {
  const fileName = String(entry.path || "").split("/").pop();
  return fileName?.toLowerCase().endsWith(".csv")
    ? fileName
    : `${safeFileComponent(entry.submission_id)}.csv`;
}
function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function fetchResult(entry) {
  const response = await fetch(`${assetUrl(entry.path)}?v=${encodeURIComponent(entry.csv_sha256 || entry.submission_id)}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${entry.path}: ${response.status}`);
  return response.arrayBuffer();
}
async function downloadEntries(entries, filename, directSingle = false) {
  if (!entries.length) throw new Error("No result CSV files match the current selection.");
  if (directSingle && entries.length === 1) {
    triggerDownload(new Blob([await fetchResult(entries[0])], { type: "text/csv;charset=utf-8" }), resultFileName(entries[0]));
    return;
  }
  const zip = new JSZip();
  for (const entry of entries) {
    const folder = [entry.backend_id, entry.dataset_id, entry.dtype, `n${entry.rhs_columns}`, entry.dense_layout, entry.method_id].map(safeFileComponent).join("/");
    zip.file(`${folder}/${resultFileName(entry)}`, await fetchResult(entry));
  }
  triggerDownload(await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } }), `${safeFileComponent(filename)}.zip`);
}
function entriesForMethod(item) { const ids = new Set(item.submissionIds); return filteredEntries.value.filter((entry) => ids.has(entry.submission_id)); }
async function downloadSubmission(item) {
  downloading.value = `result:${item.key}`; error.value = "";
  try {
    if (item.selectionRole === "best") {
      const first = item.resultRows[0];
      const submissionId = [item.backendId, item.dataset, `n${item.rhsColumns}`, item.candidateGroup || first.candidate_group, item.selectionMetric].map(safeFileComponent).join("-");
      const headers = first.csv_headers;
      if (!headers?.length) throw new Error("Source CSV headers are unavailable for this BEST selection.");
      const records = item.resultRows.map((row) => {
        const output = {
          ...row,
          submission_id: `derived-${submissionId}`,
          method_id: item.methodId,
          method_name: item.methodName,
          configuration_id: "per-matrix-best",
          candidate_group: item.candidateGroup || row.candidate_group,
          selection_role: "best",
          selection_metric: resultTimeMode(),
          selected_from: item.selectedFrom,
          base_format: "manual-selection",
          selected_configuration_id: row.selected_configuration_id || row.configuration_id,
        };
        return headers.map((header) => JSON.stringify(String(output[header] ?? ""))).join(",");
      });
      triggerDownload(new Blob([[headers.join(","), ...records].join("\n") + "\n"], { type: "text/csv" }), `${safeFileComponent(item.methodName)}-${item.backendId}-n${item.rhsColumns}.csv`);
    } else {
      await downloadEntries(entriesForMethod(item), `${item.methodName}-${item.backendId}-n${item.rhsColumns}-results`, true);
    }
  }
  catch (cause) { error.value = `Unable to download result data: ${cause.message}`; }
  finally { downloading.value = null; }
}
function safeSourcePath(value) { const path = String(value || ""); const parts = path.split("/"); if (!path || path.includes("\\") || parts.some((part) => !part || part === "." || part === "..")) throw new Error(`Unsafe source path: ${path}`); return parts.join("/"); }
async function downloadSource(item) {
  downloading.value = `source:${item.key}`; error.value = "";
  try {
    const zip = new JSZip(); const manifests = [...new Set(item.sourceManifests || [])];
    if (!manifests.length) throw new Error("No source package was published for this method.");
    for (let index = 0; index < manifests.length; index += 1) {
      const manifestPath = manifests[index]; const response = await fetch(assetUrl(manifestPath), { cache: "no-store" }); if (!response.ok) throw new Error(`${manifestPath}: ${response.status}`);
      const manifest = await response.json();
      if (!Array.isArray(manifest.files)) throw new Error(`${manifestPath} does not list source files.`);
      const selected = Array.isArray(manifest.configurations)
        ? manifest.configurations.find((configuration) => configuration.configuration_id === item.configurationId)
        : null;
      if (selected) {
        manifest.configuration_id = selected.configuration_id;
        manifest.entry_source = selected.entry_source;
      }
      const root = `plugin${index ? `-${index + 1}` : ""}`;
      zip.file(`${root}/plugin.json`, JSON.stringify(manifest, null, 2) + "\n");
      const directory = manifestPath.slice(0, manifestPath.lastIndexOf("/") + 1);
      for (const file of manifest.files.map(safeSourcePath)) {
        const sourcePath = `${directory}files/${file}`; const sourceResponse = await fetch(assetUrl(sourcePath), { cache: "no-store" });
        if (!sourceResponse.ok) throw new Error(`${sourcePath}: ${sourceResponse.status}`);
        zip.file(`${root}/${file}`, await sourceResponse.arrayBuffer());
      }
    }
    triggerDownload(await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } }), `${safeFileComponent(item.methodName)}-source.zip`);
  } catch (cause) { error.value = `Unable to download source code: ${cause.message}`; }
  finally { downloading.value = null; }
}
async function downloadAllResults() {
  downloading.value = "all"; error.value = "";
  try { await downloadEntries(downloadableEntries.value, "qiwu-spmm-all-results"); }
  catch (cause) { error.value = `Unable to download result data: ${cause.message}`; }
  finally { downloading.value = null; }
}
watch(() => [filters.backend, filters.dataset, filters.rhs], loadSelectedData);
watch(() => [filters.backend, filters.dataset, filters.rhs, filters.baseFormat, filters.timeMode, filters.iterations, rows.value.length], async () => {
  currentPage.value = 1;
  selectedMethodKey.value = ranking.value[0]?.key || "";
  await nextTick();
  drawSelected();
});
watch(ranking, (items) => {
  if (!items.some((item) => item.key === selectedMethodKey.value)) {
    selectedMethodKey.value = items[0]?.key || "";
  }
});
watch(selectedPlotItem, async () => {
  await nextTick();
  drawSelected();
});
watch(pageCount, (value) => { if (currentPage.value > value) currentPage.value = value; });
onMounted(() => { window.addEventListener("scroll", updateBackToTopVisibility, { passive: true }); updateBackToTopVisibility(); loadData(); });
onBeforeUnmount(() => {
  resultController?.abort();
  window.removeEventListener("scroll", updateBackToTopVisibility);
});
</script>

<style src="../../assets/styles/contest.css"></style>

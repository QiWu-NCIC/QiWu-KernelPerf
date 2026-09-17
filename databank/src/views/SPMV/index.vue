<template>
  <main class="spmv-page">
    <header class="page-header">
      <div>
        <p class="eyebrow">QiWu / Databank</p>
        <h1>Contest</h1>
        <p class="header-copy">Normalized performance across submitted implementations and hardware.</p>
      </div>
      <nav class="header-actions" aria-label="SpMV actions">
        <a class="action-link" :href="submissionsUrl" target="_blank" rel="noopener noreferrer">Submit code</a>
        <button class="action-button" type="button" :disabled="downloading !== null" @click="downloadAllResults">
          {{ downloading ? "Preparing..." : "Download all CSVs" }}
        </button>
      </nav>
    </header>

    <section class="control-band" aria-label="Leaderboard filters">
      <label>
        <span>Operator</span>
        <select v-model="filters.operator">
          <option v-for="value in operatorOptions" :key="value" :value="value">{{ formatOperator(value) }}</option>
        </select>
      </label>
      <label>
        <span>Hardware</span>
        <select v-model="filters.backend">
          <option v-for="value in backendOptions" :key="value" :value="value">{{ value }}</option>
        </select>
      </label>
      <label>
        <span>Dataset</span>
        <select v-model="filters.dataset">
          <option v-for="value in datasetOptions" :key="value" :value="value">{{ value }}</option>
        </select>
      </label>
      <label>
        <span>Base format</span>
        <select v-model="filters.baseFormat">
          <option value="all">All formats</option>
          <option v-for="value in baseFormatOptions" :key="value" :value="value">{{ formatBaseFormat(value) }}</option>
        </select>
      </label>
      <div class="metric-note">
        <strong>Efficiency</strong>
        <span>perf / dtype peak &times; 100%</span>
        <small>BEST uses solve-only per-matrix selection</small>
      </div>
    </section>

    <section class="status-line" aria-live="polite">
      <span>{{ displayRows.length }} result rows</span>
      <span>{{ ranking.length }} methods with measured coverage</span>
      <span v-if="loading">Loading data...</span>
      <span v-if="error" class="error-text">{{ error }}</span>
    </section>

    <section class="protocol-section" aria-labelledby="protocol-title">
      <div class="protocol-heading">
        <div>
          <p class="eyebrow">Reproducibility</p>
          <h2 id="protocol-title">Benchmark protocol</h2>
        </div>
        <span>SpMV FP32 / FP64</span>
      </div>
      <div class="protocol-grid">
        <div class="protocol-item">
          <span class="protocol-label">Timing &amp; sampling</span>
          <strong>{{ benchmarkProtocol.warmup }} warmups &middot; {{ benchmarkProtocol.iterations }} timed solves</strong>
          <p>CUDA Events (steady clock for host-timed plugins) measure solves; preprocessing uses host wall time plus stream synchronization.</p>
          <p>Warmup, validation and teardown are excluded.</p>
        </div>
        <div class="protocol-item">
          <span class="protocol-label">Correctness</span>
          <strong>Independent CPU CSR reference</strong>
          <code>&rho;<sub>i</sub> = |y<sub>i</sub> &minus; y<sub>ref,i</sub>| / s<sub>i</sub> &le; C &middot; n<sub>i</sub> &middot; u</code>
          <p><em>ref</em> is stored as host <code>long double</code> for both dtypes; products are promoted before accumulation.</p>
          <p>If <em>s<sub>i</sub></em> = 0, <em>y<sub>i</sub></em> must be zero.</p>
        </div>
        <div class="protocol-item">
          <span class="protocol-label">Terms</span>
          <div class="protocol-definitions">
            <div class="protocol-definition"><code>y<sub>i</sub></code><span>GPU row output</span></div>
            <div class="protocol-definition"><code>y<sub>ref,i</sub></code><span>CPU reference</span></div>
            <div class="protocol-definition"><code>s<sub>i</sub> = &sum;<sub>j</sub>|a<sub>ij</sub>x<sub>j</sub>|</code><span>row scale</span></div>
            <div class="protocol-definition"><code>n<sub>i</sub></code><span>row nnz count</span></div>
            <div class="protocol-definition"><code>u = &epsilon;(storage)/2</code><span>unit roundoff</span></div>
            <div class="protocol-definition"><code>C = {{ benchmarkProtocol.safetyFactor }}</code><span>safety factor</span></div>
          </div>
        </div>
        <div class="protocol-item">
          <span class="protocol-label">Code locations</span>
          <div class="protocol-paths">
            <template v-for="source in protocolSources" :key="source.path">
              <a v-if="repositoryUrl" :href="sourceFileUrl(source)" target="_blank" rel="noopener noreferrer">{{ source.label }}</a>
              <code v-else>{{ source.label }}</code>
            </template>
          </div>
        </div>
      </div>
    </section>

    <section class="summary-section" aria-labelledby="top-ranking-title">
      <div class="section-heading summary-heading">
        <div>
          <p class="eyebrow">Leading submissions</p>
          <h2 id="top-ranking-title">Leaderboard</h2>
        </div>
        <div class="summary-tools">
          <div class="ranking-mode" role="group" aria-label="Ranking time mode">
            <button
              v-for="mode in timeModes"
              :key="mode"
              type="button"
              :class="{ active: filters.timeMode === mode }"
              @click="chooseTimeMode(mode)"
            >{{ mode === "pre/iteration+solve" ? `${mode} (${filters.iterations})` : mode }}</button>
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
          <thead>
            <tr>
              <th>Rank</th>
              <th>Submission</th>
              <th>Base format</th>
              <th>Geo. mean GFLOP/s</th>
              <th>Efficiency</th>
              <th>Geo. mean preprocess</th>
              <th>Geo. mean solve</th>
              <th>Geo. mean effective</th>
              <th>Passing</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="item in pagedRanking"
              :key="item.key"
              class="ranking-row"
              role="link"
              tabindex="0"
              :aria-label="`Show scatter plots for ${item.methodName}`"
              @click="scrollToMethod(item.key)"
              @keydown.enter.prevent="scrollToMethod(item.key)"
              @keydown.space.prevent="scrollToMethod(item.key)"
            >
              <td class="table-rank">{{ item.rank ?? "Unranked" }}</td>
              <td>
                <strong>{{ item.methodName }}</strong>
                <small v-if="item.configurationId">{{ item.configurationId }}</small>
              </td>
              <td><span class="format-label">{{ formatBaseFormat(item.baseFormat) }}</span></td>
              <td>{{ formatMetricGflops(item.geomeanGflops) }}</td>
              <td>{{ formatPercent(item.score) }}%</td>
              <td>{{ formatMilliseconds(item.geomeanPreprocessMs) }}</td>
              <td>{{ formatMilliseconds(item.geomeanSolveMs) }}</td>
              <td>{{ formatMilliseconds(item.geomeanEffectiveMs) }}</td>
              <td>{{ item.passCount }}/{{ item.expectedCaseCount }}<small v-if="item.failedCount">{{ item.failedCount }} failed</small></td>
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

      <div v-if="ranking.length" class="method-bands">
        <article
          v-for="item in ranking"
          :id="methodBandId(item.key)"
          :key="item.key"
          class="method-band"
          tabindex="-1"
        >
          <header class="method-band-header">
            <div class="method-identity">
              <span class="rank-number">{{ item.rank ?? "UR" }}</span>
              <div class="method-copy">
                <strong>{{ item.methodName }}</strong>
                <span>GPU: {{ item.hardware }} &middot; CPU: {{ item.cpuModel || "not recorded" }} &middot; {{ formatBaseFormat(item.baseFormat) }} &middot; {{ item.dataset }}</span>
                <small v-if="item.configurationId">config: {{ item.configurationId }}<span v-if="item.selectionRole === 'best'"> &middot; per-matrix best</span></small>
                <small v-if="configurationDescription(item)" class="config-provenance">{{ configurationDescription(item) }}</small>
                <small v-if="item.selectedFrom">selected from: {{ item.selectedFrom }}</small>
                <small>{{ item.passCount }}/{{ item.expectedCaseCount }} passing cases &middot; {{ formatPercent(item.coverageRatio * 100) }}% coverage<span v-if="item.failedCount"> &middot; {{ item.failedCount }} failed</span></small>
              </div>
            </div>
            <div class="method-band-actions">
              <div class="efficiency-score" :aria-label="item.score > 0 ? `Efficiency ${formatPercent(item.score)} percent` : 'No passing cases'">
                <span>Efficiency</span>
                <strong v-if="item.score > 0">{{ formatPercent(item.score) }}%</strong>
                <strong v-else>n/a</strong>
                <small v-if="item.rankEligible">FP32 + FP64 geomean</small>
                <small v-else>Unranked: coverage below 90%</small>
              </div>
              <div class="download-actions">
                <button class="submission-download" type="button" :disabled="downloading !== null" @click="downloadSubmission(item)">
                  {{ downloading === `result:${item.key}` ? "Preparing..." : "Download result CSVs" }}
                </button>
                <button
                  v-if="item.sourceManifests.length"
                  class="source-download"
                  type="button"
                  :disabled="downloading !== null || !item.sourceManifests.length"
                  :title="item.sourceManifests.length ? 'Download associated source package' : 'No source package was published for this result'"
                  @click="downloadSource(item)"
                >{{ downloading === `source:${item.key}` ? "Preparing..." : "Download source" }}</button>
              </div>
            </div>
          </header>

          <div class="method-plot-grid">
            <section v-for="dtype in dtypes" :key="dtype" class="dtype-plot">
              <div class="plot-card-heading">
                <h3>{{ dtype.toUpperCase() }}</h3>
                <span>{{ methodRows(item, dtype).length }} cases &middot; GFLOP/s</span>
              </div>
              <canvas
                :ref="(element) => setCanvasRef(item.key, dtype, element)"
                class="scatter-canvas"
                :aria-label="`${item.methodName} ${dtype.toUpperCase()} GFLOP/s scatter plot`"
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
import benchmarkSpecs from "../../../../KernelPerf/config/benchmarks.json";

const spmvBenchmarkSpec = benchmarkSpecs.find((item) => item.benchmark_id === "spmv");
const spmvProtocolByDtype = Object.fromEntries((spmvBenchmarkSpec?.operators || []).map((operator) => [
  operator.dtype,
  operator.metadata?.driver_config || {},
]));

function sharedProtocolValue(field) {
  const entries = ["fp32", "fp64"].map((dtype) => [dtype, spmvProtocolByDtype[dtype]?.[field]]);
  if (entries.every(([, value]) => value === entries[0][1])) return String(entries[0][1] ?? "not configured");
  return entries.map(([dtype, value]) => `${dtype.toUpperCase()} ${value ?? "not configured"}`).join(" / ");
}

const benchmarkProtocol = {
  warmup: sharedProtocolValue("warmup"),
  iterations: sharedProtocolValue("iterations"),
  safetyFactor: sharedProtocolValue("validation_safety_factor"),
};
const repositoryUrl = String(
  import.meta.env.VITE_REPOSITORY_URL || "https://github.com/QiWu-NCIC/QiWu-KernelPerf",
).replace(/\/$/, "");
const submissionsUrl = `${repositoryUrl}/tree/main/KernelPerf/submissions`;
const protocolSources = [
  { label: "KernelPerf/config/benchmarks.json:22", path: "KernelPerf/config/benchmarks.json", line: 22 },
  { label: "KernelPerf/benchmarks/spmv/template.cu:225", path: "KernelPerf/benchmarks/spmv/template.cu", line: 225 },
  { label: "KernelPerf/benchmarks/spmv/template.cu:528", path: "KernelPerf/benchmarks/spmv/template.cu", line: 528 },
  { label: "KernelPerf/benchmarks/spmv/template.cu:563", path: "KernelPerf/benchmarks/spmv/template.cu", line: 563 },
  { label: "KernelPerf/benchmarks/spmv/template.cu:609", path: "KernelPerf/benchmarks/spmv/template.cu", line: 609 },
  { label: "KernelPerf/benchmarks/spmv/driver.py:532", path: "KernelPerf/benchmarks/spmv/driver.py", line: 532 },
];

function sourceFileUrl(source) {
  return `${repositoryUrl}/blob/main/${source.path}#L${source.line}`;
}

const rows = ref([]);
const loading = ref(false);
const error = ref("");
const downloading = ref(null);
const showBackToTop = ref(false);
const submissionEntries = ref([]);
const platformMetadata = ref({});
const dtypes = ["fp32", "fp64"];
const timeModes = ["solve-only", "pre+solve", "pre/iteration+solve"];
const canvasRefs = new Map();
const filters = reactive({ operator: "", backend: "", dataset: "", baseFormat: "all", timeMode: "solve-only", iterations: 100 });
const pageSize = 20;
const currentPage = ref(1);

function canvasKey(methodKey, dtype) {
  return `${methodKey}|${dtype}`;
}

function setCanvasRef(methodKey, dtype, element) {
  const key = canvasKey(methodKey, dtype);
  if (element) canvasRefs.set(key, element);
  else canvasRefs.delete(key);
}

function methodBandId(methodKey) {
  let hash = 2166136261;
  for (const character of String(methodKey)) {
    hash ^= character.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `method-band-${(hash >>> 0).toString(36)}`;
}

async function scrollToMethod(methodKey) {
  await nextTick();
  const target = document.getElementById(methodBandId(methodKey));
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  target.focus({ preventScroll: true });
}

function updateBackToTopVisibility() {
  showBackToTop.value = window.scrollY > 320;
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function operatorFamily(operatorId) {
  return String(operatorId || "unknown").split(".", 1)[0];
}

const operatorOptions = computed(() => [...new Set(rows.value.map((row) => operatorFamily(row.operator_id)).filter(Boolean))].sort());
const backendOptions = computed(() => [...new Set(rows.value.map((row) => row.backend_id).filter(Boolean))].sort());
const datasetOptions = computed(() => [...new Set(rows.value.map((row) => row.dataset_id).filter(Boolean))].sort());
const baseFormatOptions = computed(() => [...new Set(rows.value.map((row) => baseFormatKey(row.base_format)))].sort());
const scopedRows = computed(() => rows.value.filter((row) => {
  if (filters.operator && operatorFamily(row.operator_id) !== filters.operator) return false;
  if (filters.backend && row.backend_id !== filters.backend) return false;
  if (filters.dataset && row.dataset_id !== filters.dataset) return false;
  return true;
}));

function cusparseCsrBestName(candidates) {
  const versions = [...new Set(candidates.map((row) => (
    /^cuSPARSE CUDA ([^\s]+)/i.exec(row.method_name || "")?.[1]
  )).filter(Boolean))].sort();
  return versions.length
    ? `cuSPARSE CUDA ${versions.join("/")} CSR BEST`
    : "cuSPARSE CSR BEST";
}

// Derive the cuSPARSE CSR per-matrix winner in the browser so the contest
// shows the same solve-only selection that is used for the other BEST rows.
const displayRows = computed(() => {
  const source = scopedRows.value;
  const derived = [];
  for (const dtype of dtypes) {
    const candidates = source.filter((row) => row.dtype === dtype
      && row.candidate_group === "cusparse"
      && ["csr-default", "csr-alg1", "csr-alg2"].includes(row.configuration_id)
      && row.status === "pass" && row.solve_ms > 0);
    const byMatrix = new Map();
    const methodName = cusparseCsrBestName(candidates);
    for (const row of candidates) {
      const previous = byMatrix.get(row.matrix_id);
      if (!previous || row.solve_ms < previous.solve_ms) byMatrix.set(row.matrix_id, row);
    }
    for (const row of byMatrix.values()) {
      derived.push({
        ...row,
        method_id: "cusparse-csr-best",
        method_name: methodName,
        configuration_id: "per-matrix-best",
        candidate_group: "cusparse-csr-best",
        selection_role: "best",
        selected_from: row.configuration_id,
        base_format: "manual-selection",
      });
    }
  }
  return [...source, ...derived].filter((row) => (
    filters.baseFormat === "all" || baseFormatKey(row.base_format) === filters.baseFormat
  ));
});

const filteredSubmissionEntries = computed(() => submissionEntries.value.filter((entry) => {
  if (filters.operator && operatorFamily(entry.operator_id) !== filters.operator) return false;
  if (filters.backend && entry.backend_id !== filters.backend) return false;
  if (filters.dataset && (entry.dataset_id || "none") !== filters.dataset) return false;
  if (filters.baseFormat !== "all" && baseFormatKey(inferredBaseFormat(
    entry.base_format,
    entry.configuration_id,
    entry.selection_role || "candidate",
  )) !== filters.baseFormat) return false;
  return true;
}));

function selectFirstAvailableFilter() {
  if (!operatorOptions.value.includes(filters.operator)) filters.operator = operatorOptions.value[0] || "";
  if (!backendOptions.value.includes(filters.backend)) filters.backend = backendOptions.value[0] || "";
  if (!datasetOptions.value.includes(filters.dataset)) filters.dataset = datasetOptions.value[0] || "";
}

function parseCsv(text) {
  const result = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(value);
      if (row.some((cell) => cell.trim() !== "")) result.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }
  if (value || row.length) {
    row.push(value);
    if (row.some((cell) => cell.trim() !== "")) result.push(row);
  }
  return result;
}

function normalizeRows(text, entry) {
  const parsed = parseCsv(text);
  if (parsed.length < 2) return [];
  const headers = parsed[0];
  return parsed.slice(1).map((cells) => {
    const row = Object.fromEntries(headers.map((header, index) => [header, cells[index] || ""]));
    const configurationId = row.configuration_id || entry.configuration_id || "";
    const selectionRole = row.selection_role || entry.selection_role || "candidate";
    return {
      ...row,
      submission_id: row.submission_id || entry.submission_id,
      method_id: row.method_id || entry.method_id,
      method_name: row.method_name || entry.method_name,
      configuration_id: configurationId,
      candidate_group: row.candidate_group || entry.candidate_group || "",
      selection_role: selectionRole,
      selected_from: entry.selected_from || row.selected_from || "",
      base_format: inferredBaseFormat(
        selectionRole === "best"
          ? (entry.base_format || row.base_format || "unknown")
          : (row.base_format || entry.base_format || "unknown"),
        configurationId,
        selectionRole,
      ),
      backend_id: row.backend_id || entry.backend_id,
      hardware: row.hardware || entry.hardware || entry.backend_id,
      cpu_model: row.cpu_model || entry.cpu_model || platformMetadata.value[row.backend_id || entry.backend_id]?.cpu_model || "",
      dtype: row.dtype || entry.dtype,
      operator_id: row.operator_id || entry.operator_id,
      dataset_id: row.dataset_id || entry.dataset_id || "none",
      source_manifest: entry.source_manifest || "",
      submission_path: entry.path || "",
      status: row.status || "pass",
      nnz: Number(row.nnz || 0),
      operations: Number(row.operations || (2 * Number(row.nnz || 0))),
      preprocess_ms: Number(row.preprocess_ms || 0),
      solve_ms: Number(row.solve_ms || row.runtime_ms || 0),
      peak_gflops: Number(row.peak_gflops || entry.peak_gflops || 0),
    };
  });
}

async function loadData() {
  loading.value = true;
  error.value = "";
  try {
    const base = import.meta.env.BASE_URL;
    const [manifestResponse, platformsResponse] = await Promise.all([
      fetch(`${base}data/index.json?v=${Date.now()}`, { cache: "no-store" }),
      fetch(`${base}data/platforms.json?v=${Date.now()}`, { cache: "no-store" }),
    ]);
    if (!manifestResponse.ok) throw new Error(`manifest: ${manifestResponse.status}`);
    if (!platformsResponse.ok) throw new Error(`platform metadata: ${platformsResponse.status}`);
    const manifest = await manifestResponse.json();
    const platforms = await platformsResponse.json();
    platformMetadata.value = Object.fromEntries(
      (platforms.platforms || []).map((platform) => [platform.backend_id, platform]),
    );
    const latest = new Map();
    (manifest.submissions || []).forEach((entry) => {
      const key = [
        entry.method_id,
        entry.configuration_id || "",
        entry.backend_id,
        operatorFamily(entry.operator_id),
        entry.dtype,
        entry.dataset_id || "none",
      ].join("|");
      const previous = latest.get(key);
      if (!previous || String(entry.created_at || "") >= String(previous.created_at || "")) {
        latest.set(key, entry);
      }
    });
    const submissions = [...latest.values()];
    submissionEntries.value = submissions;
    const loaded = await Promise.allSettled(submissions.map(async (entry) => {
      const response = await fetch(`${base}${entry.path}?v=${encodeURIComponent(entry.submission_id)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`${entry.path}: ${response.status}`);
      return normalizeRows(await response.text(), entry);
    }));
    rows.value = loaded
      .filter((result) => result.status === "fulfilled")
      .flatMap((result) => result.value);
    selectFirstAvailableFilter();
    const failures = loaded.filter((result) => result.status === "rejected");
    if (failures.length) {
      error.value = `${failures.length} result file(s) could not be loaded.`;
    }
  } catch (cause) {
    error.value = `Unable to load result data: ${cause.message}`;
  } finally {
    loading.value = false;
    await nextTick();
    drawAll();
  }
}

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
  if (row.status !== "pass" || time <= 0 || row.operations <= 0) return 0;
  return row.operations / (time * 1e6);
}

function efficiency(row) {
  const performance = performanceGflops(row);
  if (performance <= 0 || row.peak_gflops <= 0) return 0;
  return performance / row.peak_gflops * 100;
}

function methodRows(item, dtype) {
  return displayRows.value.filter((row) => (
    row.dtype === dtype
    && (row.method_id || row.method_name) === item.methodId
    && (row.configuration_id || "") === item.configurationId
    && row.backend_id === item.backendId
    && operatorFamily(row.operator_id) === item.operatorFamily
    && (row.dataset_id || "none") === item.dataset
    && performanceGflops(row) > 0
  ));
}

function geometricMean(values) {
  const valid = values.filter((value) => value > 0);
  return valid.length ? Math.exp(valid.reduce((sum, value) => sum + Math.log(value), 0) / valid.length) : 0;
}

function formatPercent(value) {
  return Number(value || 0).toPrecision(5);
}

function formatBaseFormat(value) {
  const normalized = baseFormatKey(value);
  const labels = {
    csr: "CSR",
    coo: "COO",
    csc: "CSC",
    ell: "ELL",
    sell: "SELL",
    hyb: "HYB",
    bsr: "BSR",
    dia: "DIA",
    auto: "Manual Selection",
    "manual-selection": "Manual Selection",
    unmarked: "Unmarked",
    unknown: "Unmarked",
  };
  if (normalized.startsWith("sell-")) return "SELL";
  return labels[normalized] || String(value || "Unmarked");
}

function configurationDescription(item) {
  const method = String(item.methodId || "").toLowerCase();
  if (method.includes("cusparse")) {
    const version = String(item.methodName || "").match(/CUDA\s+(\d+(?:\.\d+){1,2})/i)?.[1];
    const archiveVersion = version && version.split(".").length === 2 ? `${version}.0` : version;
    const documentation = archiveVersion
      ? `https://docs.nvidia.com/cuda/archive/${archiveVersion}/cusparse/`
      : "https://docs.nvidia.com/cuda/cusparse/";
    return `NVIDIA cuSPARSE${version ? ` | CUDA ${version}` : ""} | ${documentation}`;
  }
  if (method.includes("rocsparse")) {
    const version = String(item.methodName || "").match(/DTK\s+([\d.]+)/i)?.[1];
    return `ROCm rocSPARSE${version ? ` | DTK ${version}` : ""} | https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse`;
  }
  if (method.includes("csr5")) {
    return "CSR5 | https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/pull/13 | caff9d8";
  }
  if (method.includes("adaptive")) {
    return "CSR-Adaptive | https://github.com/clMathLibraries/clSPARSE | csrmv_adaptive.cl port";
  }
  if (method.includes("ghost")) {
    return "GHOST | https://github.com/RRZE-HPC/GHOST | commit 22a004d";
  }
  if (method.includes("alphasparse")) {
    return "AlphaSparse/Library | https://github.com/AlphaSparse/Library | commit 39734b2";
  }
  return "";
}

function inferredBaseFormat(value, configurationId, selectionRole) {
  if (selectionRole === "best") return baseFormatKey(value);
  const configuration = String(configurationId || "").toLowerCase();
  for (const format of ["csr", "coo", "csc", "sell", "ell", "hyb", "bsr", "dia"]) {
    if (configuration === format || configuration.startsWith(`${format}-`)) return format;
  }
  return baseFormatKey(value);
}

function baseFormatKey(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "auto") return "manual-selection";
  return normalized.startsWith("sell-") ? "sell" : (normalized || "unknown");
}

function formatOperator(value) {
  return String(value).toLowerCase() === "spmv" ? "SpMV" : String(value);
}

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

function formatGflops(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 1) return value.toFixed(2);
  if (value >= 0.01) return value.toFixed(3);
  return value.toExponential(1);
}

function assetUrl(path) {
  return `${import.meta.env.BASE_URL}${path}`;
}

const ranking = computed(() => {
  const grouped = new Map();
  const requiredMatrices = new Map();
  displayRows.value.forEach((row) => {
    const key = [row.backend_id, operatorFamily(row.operator_id), row.dataset_id || "none", row.dtype].join("|");
    if (!requiredMatrices.has(key)) requiredMatrices.set(key, new Set());
    requiredMatrices.get(key).add(row.matrix_id);
  });
  displayRows.value.forEach((row) => {
    const dataset = row.dataset_id || "none";
    const methodId = row.method_id || row.method_name;
    const backendId = row.backend_id;
    const family = operatorFamily(row.operator_id);
    const configurationId = row.configuration_id || "";
    const key = `${methodId}|${configurationId}|${backendId}|${family}|${dataset}`;
    if (!grouped.has(key)) grouped.set(key, { key, methodId, configurationId, backendId, operatorFamily: family, methodName: row.method_name || row.method_id, hardware: row.hardware, cpuModel: row.cpu_model || platformMetadata.value[backendId]?.cpu_model || "", baseFormat: row.base_format, candidateGroup: row.candidate_group || "", selectionRole: row.selection_role || "candidate", selectedFrom: row.selected_from || "", dataset, sourceManifests: new Set(), submissionIds: new Set(), byDtype: new Map(), matricesByDtype: new Map(), performanceValues: [], preprocessValues: [], solveValues: [], effectiveValues: [], caseCount: 0, passCount: 0, failedCount: 0 });
    const item = grouped.get(key);
    if (row.submission_id) item.submissionIds.add(row.submission_id);
    if (row.source_manifest) item.sourceManifests.add(row.source_manifest);
    item.caseCount += 1;
    if (row.status !== "pass") {
      item.failedCount += 1;
      return;
    }
    if (!item.byDtype.has(row.dtype)) item.byDtype.set(row.dtype, []);
    if (!item.matricesByDtype.has(row.dtype)) item.matricesByDtype.set(row.dtype, new Set());
    item.matricesByDtype.get(row.dtype).add(row.matrix_id);
    const value = efficiency(row);
    if (value > 0) {
      item.byDtype.get(row.dtype).push(value);
      item.performanceValues.push(performanceGflops(row));
      item.preprocessValues.push(row.preprocess_ms);
      item.solveValues.push(row.solve_ms);
      item.effectiveValues.push(effectiveMs(row));
      item.passCount += 1;
    }
  });
  return [...grouped.values()]
    .map((item) => {
      const scores = [...item.byDtype.values()].map(geometricMean).filter((value) => value > 0);
      const coverageByDtype = ["fp32", "fp64"].map((dtype) => {
        const expected = requiredMatrices.get([item.backendId, item.operatorFamily, item.dataset, dtype].join("|"));
        const actual = item.matricesByDtype.get(dtype);
        return expected?.size ? (actual?.size || 0) / expected.size : 0;
      });
      const expectedCaseCount = ["fp32", "fp64"].reduce((sum, dtype) => {
        const expected = requiredMatrices.get([item.backendId, item.operatorFamily, item.dataset, dtype].join("|"));
        return sum + (expected?.size || 0);
      }, 0);
      const coverageRatio = expectedCaseCount ? item.passCount / expectedCaseCount : 0;
      const rankEligible = coverageByDtype.every((value) => value >= 0.9) && scores.length === 2;
      return {
        ...item,
        submissionIds: [...item.submissionIds],
        sourceManifests: [...item.sourceManifests],
        score: geometricMean(scores),
        geomeanGflops: geometricMean(item.performanceValues),
        geomeanPreprocessMs: geometricMean(item.preprocessValues),
        geomeanSolveMs: geometricMean(item.solveValues),
        geomeanEffectiveMs: geometricMean(item.effectiveValues),
        coverage: scores.length,
        coverageRatio,
        coverageByDtype,
        expectedCaseCount,
        failedCount: Math.max(item.failedCount, expectedCaseCount - item.passCount),
        rankEligible,
      };
    })
    .filter((item) => item.caseCount > 0)
    .sort((left, right) => Number(right.rankEligible) - Number(left.rankEligible)
      || (right.score - left.score) || (right.passCount - left.passCount))
    .reduce((items, item) => {
      const rankedCount = items.filter((value) => value.rank !== null).length;
      items.push({ ...item, rank: item.rankEligible ? rankedCount + 1 : null });
      return items;
    }, []);
});

const pageCount = computed(() => Math.max(1, Math.ceil(ranking.value.length / pageSize)));
const pageStart = computed(() => Math.min((currentPage.value - 1) * pageSize, Math.max(0, ranking.value.length - 1)));
const pageEnd = computed(() => Math.min(pageStart.value + pageSize, ranking.value.length));
const pagedRanking = computed(() => ranking.value.slice(pageStart.value, pageEnd.value));

function drawScatter(canvas, data) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(280, rect.width || 640);
  const height = 340;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  const pad = { left: 68, right: 18, top: 20, bottom: 44 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const xValues = data.map((row) => Math.max(1, Number.isFinite(row.nnz) ? row.nnz : 1));
  const yValues = data.map(performanceGflops);
  const rawXMin = Math.min(...(xValues.length ? xValues : [1]));
  const rawXMax = Math.max(...(xValues.length ? xValues : [10]));
  const xLogMin = Math.floor(Math.log10(rawXMin));
  const xLogMax = Math.max(xLogMin + 1, Math.ceil(Math.log10(rawXMax)));
  const xMin = 10 ** xLogMin;
  const xMax = 10 ** xLogMax;
  const yMax = Math.max(...(yValues.length ? yValues : [1])) * 1.15;
  const xScale = (value) => pad.left + (Math.log10(Math.max(xMin, value)) - xLogMin) / (xLogMax - xLogMin) * plotWidth;
  const yScale = (value) => pad.top + plotHeight - Math.max(0, value) / yMax * plotHeight;
  context.strokeStyle = "#e8edf2";
  context.fillStyle = "#647180";
  context.font = "12px system-ui, sans-serif";
  for (let index = 0; index <= 4; index += 1) {
    const y = pad.top + plotHeight * index / 4;
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(pad.left + plotWidth, y); context.stroke();
    context.fillStyle = "#647180";
    context.textAlign = "right";
    context.fillText(formatGflops(yMax * (4 - index) / 4), pad.left - 8, y + 4);
  }
  const exponentSpan = xLogMax - xLogMin;
  const tickExponents = exponentSpan <= 6
    ? Array.from({ length: exponentSpan + 1 }, (_, index) => xLogMin + index)
    : Array.from({ length: 5 }, (_, index) => Math.round(xLogMin + exponentSpan * index / 4));
  tickExponents.forEach((exponent) => {
    const tick = 10 ** exponent;
    const x = xScale(tick);
    context.beginPath(); context.moveTo(x, pad.top); context.lineTo(x, pad.top + plotHeight); context.stroke();
    context.textAlign = "center";
    context.fillText(formatNnz(tick), x, height - 27);
  });
  context.strokeStyle = "#aeb9c5";
  context.beginPath(); context.moveTo(pad.left, pad.top); context.lineTo(pad.left, pad.top + plotHeight); context.lineTo(pad.left + plotWidth, pad.top + plotHeight); context.stroke();
  context.textAlign = "center"; context.fillText("matrix nnz (log)", pad.left + plotWidth / 2, height - 12);
  context.save(); context.translate(14, pad.top + plotHeight / 2); context.rotate(-Math.PI / 2); context.fillText("GFLOP/s", 0, 0); context.restore();
  data.forEach((row) => {
    context.fillStyle = "rgba(37, 77, 158, 0.48)";
    context.beginPath(); context.arc(xScale(row.nnz), yScale(performanceGflops(row)), 4, 0, Math.PI * 2); context.fill();
  });
  if (!data.length) {
    context.fillStyle = "#647180"; context.textAlign = "center"; context.fillText("No published cases", pad.left + plotWidth / 2, pad.top + plotHeight / 2);
  }
}

function formatNnz(value) {
  if (value >= 1e6) return `${(value / 1e6).toPrecision(3)}M`;
  if (value >= 1e3) return `${(value / 1e3).toPrecision(3)}K`;
  return Math.round(value).toString();
}

function drawAll() {
  ranking.value.forEach((item) => {
    dtypes.forEach((dtype) => {
      drawScatter(canvasRefs.get(canvasKey(item.key, dtype)), methodRows(item, dtype));
    });
  });
}

function safeFileComponent(value, fallback = "result") {
  const normalized = String(value || "")
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || fallback;
}

function csvFileName(entry) {
  const fileName = String(entry.path || "").split("/").pop();
  return fileName?.toLowerCase().endsWith(".csv")
    ? fileName
    : `${safeFileComponent(entry.submission_id)}.csv`;
}

function triggerDownload(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function fetchResultFile(entry) {
  const response = await fetch(`${assetUrl(entry.path)}?v=${encodeURIComponent(entry.submission_id)}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${entry.path}: ${response.status}`);
  return response.arrayBuffer();
}

async function downloadEntries(entries, archiveName, { directSingle = false } = {}) {
  if (!entries.length) throw new Error("No result CSV files match the current selection.");
  if (directSingle && entries.length === 1) {
    const content = await fetchResultFile(entries[0]);
    triggerDownload(new Blob([content], { type: "text/csv;charset=utf-8" }), csvFileName(entries[0]));
    return;
  }
  const zip = new JSZip();
  for (const entry of entries) {
    const format = inferredBaseFormat(
      entry.base_format,
      entry.configuration_id,
      entry.selection_role || "candidate",
    );
    const directory = [
      operatorFamily(entry.operator_id),
      entry.backend_id,
      entry.dataset_id || "none",
      format,
      entry.dtype || "unknown",
    ].map((value) => safeFileComponent(value)).join("/");
    zip.file(`${directory}/${csvFileName(entry)}`, await fetchResultFile(entry));
  }
  const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } });
  triggerDownload(blob, `${safeFileComponent(archiveName)}.zip`);
}

function entriesForSubmission(item) {
  const submissionIds = new Set(item.submissionIds || []);
  return submissionEntries.value.filter((entry) => submissionIds.has(entry.submission_id));
}

async function downloadSubmission(item) {
  downloading.value = `result:${item.key}`;
  error.value = "";
  try {
    const name = `${item.methodName}-${item.configurationId || "default"}-${item.backendId}-${item.dataset}`;
    await downloadEntries(entriesForSubmission(item), name, { directSingle: true });
  } catch (cause) {
    error.value = `Unable to download result data: ${cause.message}`;
  } finally {
    downloading.value = null;
  }
}

function safeSourcePath(value) {
  const path = String(value || "");
  const parts = path.split("/");
  if (!path || path.includes("\\") || parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error(`Source manifest contains an unsafe path: ${path || "<empty>"}`);
  }
  return parts.join("/");
}

async function downloadSource(item) {
  downloading.value = `source:${item.key}`;
  error.value = "";
  try {
    const entries = entriesForSubmission(item).filter((entry) => entry.source_manifest);
    if (!entries.length) throw new Error("No source plugin was published for this result.");
    const zip = new JSZip();
    const seen = new Set();
    for (const entry of entries) {
      if (seen.has(entry.source_manifest)) continue;
      seen.add(entry.source_manifest);
      const manifestResponse = await fetch(assetUrl(entry.source_manifest), { cache: "no-store" });
      if (!manifestResponse.ok) throw new Error(`${entry.source_manifest}: ${manifestResponse.status}`);
      const publishedManifestText = await manifestResponse.text();
      const manifest = JSON.parse(publishedManifestText);
      if (!Array.isArray(manifest.files)) throw new Error(`${entry.source_manifest} does not list source files.`);
      const selected = Array.isArray(manifest.configurations)
        ? manifest.configurations.find((configuration) => configuration.configuration_id === item.configurationId)
        : null;
      if (selected) {
        manifest.configuration_id = selected.configuration_id;
        manifest.entry_source = selected.entry_source;
      }
      const manifestText = JSON.stringify(manifest, null, 2) + "\n";
      const root = seen.size === 1 ? "plugin" : `plugin-${seen.size}`;
      zip.file(`${root}/plugin.json`, manifestText);
      const manifestDirectory = entry.source_manifest.slice(0, entry.source_manifest.lastIndexOf("/") + 1);
      const sourceFiles = manifest.files.map((file) => safeSourcePath(file));
      for (let offset = 0; offset < sourceFiles.length; offset += 16) {
        const batch = sourceFiles.slice(offset, offset + 16);
        const downloaded = await Promise.all(batch.map(async (relative) => {
          const sourcePath = `${manifestDirectory}files/${relative}`;
          const response = await fetch(assetUrl(sourcePath), { cache: "no-store" });
          if (!response.ok) throw new Error(`${sourcePath}: ${response.status}`);
          return { relative, content: await response.arrayBuffer() };
        }));
        for (const file of downloaded) zip.file(`${root}/${file.relative}`, file.content);
      }
    }
    const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } });
    const name = `${item.methodName}-${item.configurationId || "default"}-${item.backendId}-source`;
    triggerDownload(blob, `${safeFileComponent(name)}.zip`);
  } catch (cause) {
    error.value = `Unable to download source code: ${cause.message}`;
  } finally {
    downloading.value = null;
  }
}

async function downloadAllResults() {
  downloading.value = "all";
  error.value = "";
  try {
    const name = `${filters.operator || "operator"}-${filters.backend || "platform"}-${filters.dataset || "dataset"}-results`;
    await downloadEntries(filteredSubmissionEntries.value, name);
  } catch (cause) {
    error.value = `Unable to download result data: ${cause.message}`;
  } finally {
    downloading.value = null;
  }
}

watch(() => [filters.operator, filters.backend, filters.dataset, filters.baseFormat, filters.timeMode, rows.value.length], async () => {
  currentPage.value = 1;
  await nextTick();
  drawAll();
});
watch(pageCount, (value) => {
  if (currentPage.value > value) currentPage.value = value;
});
onMounted(() => {
  window.addEventListener("scroll", updateBackToTopVisibility, { passive: true });
  updateBackToTopVisibility();
  loadData();
});
onBeforeUnmount(() => window.removeEventListener("scroll", updateBackToTopVisibility));
</script>

<style scoped>
.spmv-page {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  min-height: 100vh;
  padding: 40px 10% 64px;
  overflow-x: hidden;
  background: #f5f7fb;
  color: #202d3d;
  font-family: "Microsoft YaHei", Arial, sans-serif;
}

.spmv-page,
.spmv-page * {
  box-sizing: border-box;
  letter-spacing: 0;
}

.page-header,
.control-band,
.status-line,
.protocol-section,
.summary-section,
.leaderboard-content {
  min-width: 0;
  max-width: 1440px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 32px;
  padding-bottom: 26px;
}

.page-header > div,
.header-copy,
.control-band label,
.protocol-heading > div {
  min-width: 0;
  max-width: 100%;
}

.eyebrow {
  margin: 0 0 7px;
  color: #64748a;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  max-width: 100%;
  margin-bottom: 8px;
  color: #17263a;
  font-size: 32px;
  line-height: 1.2;
  white-space: normal;
  overflow-wrap: anywhere;
}

.header-copy {
  margin-bottom: 0;
  color: #66758a;
  font-size: 14px;
  line-height: 1.6;
  overflow-wrap: anywhere;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.action-link,
.action-button {
  min-height: 38px;
  padding: 9px 14px;
  border: 1px solid #aebdd0;
  border-radius: 4px;
  background: #ffffff;
  color: #254d9e;
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
}

.action-button {
  border-color: #0a55d5;
  background: #0a55d5;
  color: #ffffff;
}

.action-link:hover {
  border-color: #0a55d5;
  color: #0a55d5;
}

.action-button:hover {
  border-color: #0848b4;
  background: #0848b4;
}

.action-button:disabled,
.submission-download:disabled {
  cursor: wait;
  opacity: .58;
}

.source-download:disabled {
  cursor: not-allowed;
  opacity: .58;
}

.control-band {
  display: grid;
  grid-template-columns: repeat(4, minmax(140px, 1fr)) minmax(150px, auto);
  align-items: end;
  gap: 14px;
  padding: 17px;
  border: 1px solid #d8e0ea;
  border-radius: 4px;
  background: #ffffff;
}

.control-band label {
  display: grid;
  gap: 6px;
  color: #526276;
  font-size: 12px;
  font-weight: 500;
}

select {
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  border: 1px solid #b9c7d6;
  border-radius: 4px;
  background: #ffffff;
  color: #202d3d;
  font: inherit;
}

select:focus {
  border-color: #0a55d5;
  outline: 2px solid rgba(10, 85, 213, .14);
  outline-offset: 1px;
}

.metric-note {
  display: grid;
  gap: 4px;
  padding: 4px 8px;
  color: #526276;
  font-size: 12px;
}

.metric-note strong {
  color: #202d3d;
}

.status-line {
  display: flex;
  gap: 22px;
  padding: 12px 2px 20px;
  color: #66758a;
  font-size: 12px;
  flex-wrap: wrap;
}

.error-text {
  color: #c34d49;
}

.protocol-section {
  margin-bottom: 34px;
  padding: 20px 0;
  border-top: 1px solid #d8e0ea;
  border-bottom: 1px solid #d8e0ea;
}

.protocol-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 18px;
}

.protocol-heading h2 {
  margin-bottom: 0;
  font-size: 20px;
}

.protocol-heading > span {
  color: #66758a;
  font-size: 12px;
  font-weight: 600;
}

.protocol-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.protocol-item {
  min-width: 0;
  padding: 0 20px;
  border-left: 1px solid #d8e0ea;
}

.protocol-item:first-child {
  padding-left: 0;
  border-left: 0;
}

.protocol-item:last-child {
  padding-right: 0;
}

.protocol-label {
  display: block;
  margin-bottom: 7px;
  color: #64748a;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}

.protocol-item strong {
  display: block;
  margin-bottom: 8px;
  color: #17263a;
  font-size: 13px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.protocol-item p {
  margin-bottom: 8px;
  color: #526276;
  font-size: 12px;
  line-height: 1.6;
  overflow-wrap: anywhere;
}

.protocol-item > code,
.protocol-paths code,
.protocol-paths a {
  color: #254d9e;
  font-family: Consolas, "Courier New", monospace;
  font-size: 10px;
  overflow-wrap: anywhere;
}

.protocol-item > code {
  display: block;
  margin: 2px 0 8px;
  line-height: 1.45;
}

.protocol-definitions {
  display: grid;
  gap: 5px;
}

.protocol-definition {
  display: grid;
  grid-template-columns: minmax(68px, auto) 1fr;
  align-items: baseline;
  gap: 8px;
  color: #526276;
  font-size: 11px;
  line-height: 1.35;
}

.protocol-definition code {
  color: #254d9e;
  font-family: Consolas, "Courier New", monospace;
  font-size: 10px;
  white-space: nowrap;
}

.protocol-paths {
  display: grid;
  gap: 7px;
  padding-top: 1px;
}

.protocol-paths a:hover {
  color: #0a55d5;
}

.leaderboard-content {
  min-width: 0;
}

.summary-section {
  min-width: 0;
  margin-bottom: 34px;
}

.summary-heading {
  align-items: flex-end;
  margin-bottom: 14px;
}

.summary-tools {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  min-width: 0;
  flex-wrap: wrap;
}

.ranking-mode {
  display: inline-grid;
  grid-template-columns: repeat(3, minmax(105px, 1fr));
  overflow: hidden;
  border: 1px solid #b9c7d6;
  border-radius: 4px;
  background: #ffffff;
}

.ranking-mode button {
  min-height: 36px;
  padding: 8px 12px;
  border: 0;
  border-left: 1px solid #d8e0ea;
  background: transparent;
  color: #526276;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}

.ranking-mode button:first-child {
  border-left: 0;
}

.ranking-mode button:hover {
  background: #f1f5fb;
}

.ranking-mode button.active {
  background: #254d9e;
  color: #ffffff;
  font-weight: 600;
}

.pagination {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #526276;
  font-size: 12px;
  white-space: nowrap;
}

.pagination button {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  padding: 0;
  border: 1px solid #b9c7d6;
  border-radius: 4px;
  background: #ffffff;
  color: #254d9e;
  font: inherit;
  font-size: 17px;
  line-height: 1;
  cursor: pointer;
}

.pagination button:hover:not(:disabled) {
  border-color: #254d9e;
  background: #f2f6fd;
}

.pagination button:disabled {
  color: #aeb9c5;
  cursor: not-allowed;
}

.pagination button:focus-visible {
  outline: 2px solid #256b87;
  outline-offset: 2px;
}

.table-summary {
  margin: 0 0 8px;
  color: #66758a;
  font-size: 11px;
  text-align: right;
}

.ranking-table-scroll {
  overflow-x: auto;
  border: 1px solid #d8e0ea;
  border-radius: 4px;
  background: #ffffff;
}

.ranking-table {
  width: 100%;
  min-width: 1060px;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}

.ranking-table th,
.ranking-table td {
  padding: 12px 14px;
  border-bottom: 1px solid #e6ebf1;
  text-align: right;
  vertical-align: middle;
  white-space: nowrap;
}

.ranking-table th {
  background: #f7f9fc;
  color: #526276;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}

.ranking-table th:nth-child(2),
.ranking-table td:nth-child(2),
.ranking-table th:nth-child(3),
.ranking-table td:nth-child(3) {
  text-align: left;
}

.submission-download,
.source-download {
  min-height: 30px;
  padding: 6px 10px;
  border: 1px solid #aebdd0;
  border-radius: 4px;
  background: #ffffff;
  color: #254d9e;
  font: inherit;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}

.submission-download:hover:not(:disabled),
.source-download:hover:not(:disabled) {
  border-color: #254d9e;
  background: #f2f6fd;
}

.ranking-table tbody tr:last-child td {
  border-bottom: 0;
}

.ranking-table tbody tr:hover {
  background: #f8faff;
}

.ranking-row {
  cursor: pointer;
}

.ranking-row:focus-visible {
  background: #eef4f8;
  outline: 2px solid #256b87;
  outline-offset: -2px;
}

.ranking-table td {
  color: #344257;
  font-size: 12px;
}

.ranking-table td strong,
.ranking-table td small {
  display: block;
}

.ranking-table td strong {
  max-width: 260px;
  overflow: hidden;
  color: #202d3d;
  text-overflow: ellipsis;
}

.ranking-table td small {
  max-width: 260px;
  margin-top: 3px;
  overflow: hidden;
  color: #728095;
  text-overflow: ellipsis;
}

.table-rank {
  color: #254d9e !important;
  font-size: 16px !important;
  font-weight: 700;
  text-align: center !important;
}

.format-label {
  display: inline-block;
  padding: 3px 7px;
  border: 1px solid #cad5e2;
  border-radius: 3px;
  background: #f6f8fb;
  color: #43546a;
  font-size: 11px;
  font-weight: 600;
}

.section-heading {
  display: flex;
  max-width: 100%;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.section-heading h2 {
  margin-bottom: 0;
  font-size: 19px;
}

.score-unit {
  color: #728095;
  font-size: 11px;
  white-space: nowrap;
}

.leaderboard-heading {
  margin-bottom: 14px;
}

.method-bands {
  display: grid;
  gap: 18px;
}

.method-band {
  min-width: 0;
  padding: 18px 20px 20px;
  border: 1px solid #d8e0ea;
  border-radius: 4px;
  background: #ffffff;
  box-shadow: 0 2px 5px rgba(31, 52, 78, .08);
}

.method-band:focus {
  outline: none;
}

.method-band-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding-bottom: 14px;
  border-bottom: 1px solid #e3e9f0;
}

.method-identity {
  display: grid;
  grid-template-columns: 32px minmax(0, 1fr);
  gap: 12px;
  min-width: 0;
}

.rank-number {
  color: #254d9e;
  font-size: 19px;
  font-weight: 700;
}

.method-copy {
  display: grid;
  gap: 3px;
  min-width: 0;
}

.method-copy strong {
  overflow-wrap: anywhere;
}

.method-copy span,
.method-copy small {
  color: #66758a;
  font-size: 12px;
}

.method-copy .config-provenance {
  overflow-wrap: anywhere;
}

.method-copy a {
  width: fit-content;
  color: #254d9e;
  font-size: 12px;
}

.method-band-actions {
  display: grid;
  flex: 0 0 auto;
  justify-items: end;
  gap: 9px;
}

.download-actions {
  display: flex;
  justify-content: flex-end;
  gap: 7px;
  flex-wrap: wrap;
}

.efficiency-score {
  display: grid;
  flex: 0 0 auto;
  justify-items: end;
  gap: 2px;
  text-align: right;
}

.efficiency-score span,
.efficiency-score small {
  color: #66758a;
  font-size: 11px;
}

.efficiency-score strong {
  color: #254d9e;
  font-size: 22px;
  line-height: 1.1;
}

.empty-state {
  margin: 28px 0 0;
  color: #66758a;
  font-size: 13px;
}

.method-plot-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.dtype-plot {
  min-width: 0;
  padding: 16px 18px 0 0;
}

.dtype-plot + .dtype-plot {
  padding-right: 0;
  padding-left: 18px;
  border-left: 1px solid #e3e9f0;
}

.plot-card-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.plot-card-heading h3 {
  margin-bottom: 0;
  font-size: 16px;
}

.plot-card-heading span {
  color: #728095;
  font-size: 11px;
}

.scatter-canvas {
  display: block;
  width: 100%;
  min-height: 280px;
  margin-top: 8px;
}

.back-to-top {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 10;
  display: grid;
  width: 42px;
  height: 42px;
  place-items: center;
  padding: 0;
  border: 1px solid #aebdd0;
  border-radius: 50%;
  background: #ffffff;
  color: #254d9e;
  font: inherit;
  font-size: 22px;
  line-height: 1;
  cursor: pointer;
  box-shadow: 0 3px 10px rgba(31, 52, 78, .16);
}

.back-to-top:hover {
  border-color: #254d9e;
  background: #f2f6fd;
}

.back-to-top:focus-visible {
  outline: 2px solid #256b87;
  outline-offset: 2px;
}

@media (max-width: 1050px) {
  .spmv-page {
    padding-right: 24px;
    padding-left: 24px;
  }

  .control-band {
    grid-template-columns: repeat(2, minmax(160px, 1fr));
  }

  .protocol-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 22px 0;
  }

  .protocol-item:nth-child(3) {
    padding-left: 0;
    border-left: 0;
  }
}

@media (max-width: 750px) {
  .spmv-page {
    padding: 24px 14px 44px;
  }

  .back-to-top {
    right: 16px;
    bottom: 16px;
  }

  .page-header {
    display: grid;
    min-width: 0;
    align-items: start;
  }

  h1 {
    font-size: 27px;
  }

  .control-band,
  .method-plot-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .summary-heading {
    align-items: flex-start;
  }

  .summary-tools {
    width: 100%;
    justify-content: flex-start;
  }

  .protocol-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 8px;
  }

  .protocol-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .protocol-item,
  .protocol-item:first-child,
  .protocol-item:nth-child(3),
  .protocol-item:last-child {
    padding: 0 0 18px;
    border-bottom: 1px solid #d8e0ea;
    border-left: 0;
  }

  .protocol-item:last-child {
    padding-bottom: 0;
    border-bottom: 0;
  }

  .ranking-mode {
    width: 100%;
    grid-template-columns: 1fr;
  }

  .ranking-mode button,
  .ranking-mode button:first-child {
    border-top: 1px solid #d8e0ea;
    border-left: 0;
  }

  .ranking-mode button:first-child {
    border-top: 0;
  }

  .leaderboard-content,
  .method-band {
    max-width: 100%;
  }

  .section-heading {
    flex-wrap: wrap;
  }

  .score-unit {
    white-space: normal;
  }

  .method-band {
    padding: 14px;
  }

  .method-band-header {
    gap: 12px;
  }

  .dtype-plot {
    padding: 14px 0 0;
  }

  .dtype-plot + .dtype-plot {
    margin-top: 14px;
    padding-left: 0;
    border-top: 1px solid #e3e9f0;
    border-left: 0;
  }

  .efficiency-score strong {
    font-size: 18px;
  }
}
</style>

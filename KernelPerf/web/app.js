const state = {
  jobs: [],
  backends: [],
  suites: [],
  results: [],
  leaderboard: { enabled: false, public_url: "" },
};

const byId = (id) => document.getElementById(id);
const selected = (id) => Array.from(byId(id).selectedOptions).map((option) => option.value);

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(url + ": " + response.status + " " + await response.text());
  }
  return response.json();
}

function fillSelect(id, values, labelFn) {
  const select = byId(id);
  const chosen = new Set(selected(id));
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value.id;
    option.textContent = labelFn(value);
    option.selected = chosen.size === 0 || chosen.has(value.id);
    select.appendChild(option);
  });
}

async function refreshMetadata() {
  const [jobs, backends, suites, workers, queue, leaderboard] = await Promise.all([
    getJson("/api/v1/jobs"),
    getJson("/api/v1/backends"),
    getJson("/api/v1/suites"),
    getJson("/api/v1/workers"),
    getJson("/api/v1/queue"),
    getJson("/api/v1/leaderboard/config"),
  ]);
  state.jobs = jobs;
  state.backends = backends;
  state.suites = suites;
  state.leaderboard = leaderboard;
  fillSelect(
    "jobFilter",
    jobs.map((job) => ({ id: job.job_id, status: job.status })),
    (job) => job.id.slice(0, 8) + " " + job.status,
  );
  fillSelect(
    "backendFilter",
    backends.map((backend) => ({ id: backend.backend_id, name: backend.name })),
    (backend) => backend.name,
  );
  fillSelect(
    "suiteFilter",
    suites.map((suite) => ({ id: suite.suite_id, name: suite.display_name })),
    (suite) => suite.name,
  );
  byId("workers").innerHTML = workers.map((worker) =>
    '<div class="worker"><strong>' + worker.worker_id + '</strong><br />' +
    '<span class="muted">' + worker.backend_id + " | " + worker.status + '</span><br />' +
    "<span>" + (worker.current_job_id || "idle") + "</span></div>"
  ).join("");
  byId("queue").textContent =
    "Queued " + queue.queued_job_ids.length + " | Running " + queue.running.length;
  if (leaderboard.public_url) byId("leaderboardLink").href = leaderboard.public_url;
}

function outcomeSummary(rows) {
  const order = ["pass", "error", "fail"];
  const counts = Object.fromEntries(order.map((status) => [status, 0]));
  rows.forEach((row) => {
    const status = order.includes(row.status) ? row.status : "pass";
    counts[status] += 1;
  });
  const total = rows.length;
  const rawUnits = order.map((status) => total ? counts[status] * 1000 / total : 0);
  const units = rawUnits.map(Math.floor);
  let remainder = total ? 1000 - units.reduce((sum, value) => sum + value, 0) : 0;
  const fractions = rawUnits
    .map((value, index) => ({ index, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction || a.index - b.index);
  for (let index = 0; index < remainder; index += 1) {
    units[fractions[index].index] += 1;
  }
  return order.map((status, index) => ({
    status,
    count: counts[status],
    percent: units[index] / 10,
  }));
}

function formatBaseFormat(value) {
  const normalized = String(value || "").trim().toLowerCase();
  const labels = {
    csr: "CSR",
    coo: "COO",
    ell: "ELL",
    sell: "SELL",
    hyb: "HYB",
    bsr: "BSR",
    dia: "DIA",
    auto: "Auto-tuned",
    "auto-tuned": "Auto-tuned",
    unmarked: "Unmarked",
    unknown: "Unmarked",
  };
  if (normalized.startsWith("sell-")) return "SELL";
  return labels[normalized] || String(value || "Unmarked");
}

function formatGflops(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 1) return value.toFixed(2);
  if (value >= 0.01) return value.toFixed(3);
  return value.toExponential(1);
}

function drawPlot(canvas, rows) {
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  context.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;
  const pad = { left: 76, right: 16, top: 36, bottom: 44 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  const summary = outcomeSummary(rows);
  const outcomeColors = { pass: "#2e8b57", error: "#d58a16", fail: "#c94b4b" };
  let barX = pad.left;
  summary.forEach(({ status, percent }) => {
    const segmentWidth = plotWidth * percent / 100;
    context.fillStyle = outcomeColors[status];
    context.fillRect(barX, 10, segmentWidth, 10);
    barX += segmentWidth;
  });

  const passedRows = rows.filter((row) =>
    (row.status || "pass") === "pass" && Number(row.gflops) > 0
  );
  const xValues = passedRows.length
    ? passedRows.map((row) => Math.max(1, Number(row.nnz)))
    : [1];
  const yValues = passedRows.length
    ? passedRows.map((row) => Number(row.gflops))
    : [0];
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMax = Math.max(...yValues) * 1.15;
  const xScale = (value) => {
    const current = Math.log10(Math.max(1, value));
    const minimum = Math.log10(Math.max(1, xMin));
    const maximum = Math.log10(Math.max(10, xMax));
    return pad.left + (current - minimum) / Math.max(0.001, maximum - minimum) * plotWidth;
  };
  const yScale = (value) =>
    height - pad.bottom - Math.max(0, value) / yMax * plotHeight;

  context.strokeStyle = "#edf1f5";
  context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const gridX = pad.left + index / 4 * plotWidth;
    const gridY = pad.top + index / 4 * plotHeight;
    context.beginPath();
    context.moveTo(gridX, pad.top);
    context.lineTo(gridX, height - pad.bottom);
    context.stroke();
    context.beginPath();
    context.moveTo(pad.left, gridY);
    context.lineTo(width - pad.right, gridY);
    context.stroke();
  }

  context.strokeStyle = "#c8d0da";
  context.beginPath();
  context.moveTo(pad.left, pad.top);
  context.lineTo(pad.left, height - pad.bottom);
  context.lineTo(width - pad.right, height - pad.bottom);
  context.stroke();

  context.fillStyle = "#526070";
  context.font = "12px system-ui";
  context.textBaseline = "middle";
  context.save();
  context.translate(14, pad.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("GFLOP/s", 0, 0);
  context.restore();
  context.textAlign = "center";
  context.fillText("matrix nnz (log)", width / 2, height - 10);
  context.textAlign = "right";
  context.fillText("0", pad.left - 10, height - pad.bottom);
  context.fillText(formatGflops(yMax), pad.left - 10, pad.top);
  context.textAlign = "left";
  context.textBaseline = "alphabetic";
  context.fillText(xMin.toExponential(1), pad.left, height - pad.bottom + 18);
  context.fillText(xMax.toExponential(1), width - pad.right - 68, height - pad.bottom + 18);

  const dtype = passedRows[0] && passedRows[0].dtype;
  context.fillStyle = dtype === "fp64" ? "#b64b3f" : "#2563a6";
  passedRows.forEach((row) => {
    context.beginPath();
    context.arc(
      xScale(Number(row.nnz)),
      yScale(Number(row.gflops)),
      4,
      0,
      Math.PI * 2,
    );
    context.fill();
  });
  if (passedRows.length === 0) {
    context.fillStyle = "#667789";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(
      "No passed performance cases",
      pad.left + plotWidth / 2,
      pad.top + plotHeight / 2,
    );
  }
}

function resultParams(jobId, backendId, suite, operatorId, configurationId = "") {
  const params = new URLSearchParams({
    job_id: jobId,
    backend_id: backendId,
    suite,
    operator_id: operatorId,
  });
  if (configurationId) params.set("configuration_id", configurationId);
  return params;
}

function downloadPlotLogs(filename, rows) {
  const params = new URLSearchParams();
  params.append("job_id", rows[0].job_id);
  params.set("backend_id", rows[0].backend_id);
  params.set("suite", rows[0].suite);
  params.set("operator_id", rows[0].operator_id);
  const link = document.createElement("a");
  link.href = "/api/v1/plot-logs/download?" + params.toString();
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function actionButton(label, title, handler) {
  const button = document.createElement("button");
  button.className = "download-button";
  button.textContent = label;
  button.title = title;
  button.addEventListener("click", handler);
  return button;
}

async function publishResult(button, jobId, backendId, suite, operatorId, configurationId = "") {
  button.disabled = true;
  button.textContent = "...";
  try {
    const response = await getJson("/api/v1/leaderboard/submissions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: jobId,
        backend_id: backendId,
        suite,
        operator_id: operatorId,
        ...(configurationId ? { configuration_id: configurationId } : {}),
      }),
    });
    button.textContent = "DONE";
    button.title = response.result_path || "Published";
  } catch (error) {
    button.disabled = false;
    button.textContent = "ERR";
    button.title = error.message;
  }
}

function renderPlots(results) {
  const plots = byId("plots");
  plots.innerHTML = "";
  const groups = new Map();
  results.forEach((result) => {
    const key = JSON.stringify([
      result.job_id,
      result.backend_id,
      result.suite,
      result.operator_id,
      result.kernel_name,
      result.configuration_id || "",
    ]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(result);
  });
  if (groups.size === 0) {
    plots.innerHTML = '<div class="muted">No performance results match the selected filters.</div>';
    return;
  }

  for (const [key, rows] of groups.entries()) {
    const [jobId, backendId, suite, operatorId, kernelName, configurationId] = JSON.parse(key);
    const dtype = rows[0].dtype || "unknown";
    const baseFormat = formatBaseFormat(rows[0].base_format || "unknown");
    const job = state.jobs.find((item) => item.job_id === jobId);
    const complete = Boolean(job && job.status === "succeeded");
    const filename = [kernelName, backendId, operatorId].join("_").replace(/[^A-Za-z0-9._-]+/g, "_");
    const params = resultParams(jobId, backendId, suite, operatorId, configurationId);

    const container = document.createElement("div");
    container.className = "plot";
    const head = document.createElement("div");
    head.className = "plot-head";
    const title = document.createElement("div");
    title.className = "plot-title";
    title.textContent =
      kernelName + " | " + dtype + " | " + baseFormat + " | " +
      (configurationId ? "config=" + configurationId + " | " : "") +
      (rows[0].dataset_id || "none") + " | " + backendId +
      " | " + jobId.slice(0, 8);
    const actions = document.createElement("div");
    actions.className = "plot-actions";

    const csvButton = actionButton("CSV", "Download canonical leaderboard CSV", () => {
      window.location.href = "/api/v1/results.csv?" + params.toString();
    });
    csvButton.disabled = !complete;
    const logButton = actionButton(
      "LOG",
      "Download logs for this result",
      () => downloadPlotLogs(filename + ".log", rows),
    );
    const rankButton = actionButton("RANK", "Submit this result to the GitHub leaderboard", () => {
      publishResult(rankButton, jobId, backendId, suite, operatorId, configurationId);
    });
    rankButton.disabled = !complete || !state.leaderboard.enabled;
    if (!state.leaderboard.enabled) {
      rankButton.title = "Leaderboard publishing is not configured on this server";
    }
    actions.append(logButton, csvButton, rankButton);
    head.append(title, actions);

    const outcomes = document.createElement("div");
    outcomes.className = "plot-outcomes";
    outcomes.setAttribute("aria-label", "Case outcome rates");
    outcomeSummary(rows).forEach(({ status, count, percent }) => {
      const outcome = document.createElement("span");
      outcome.className = "plot-outcome plot-outcome-" + status;
      outcome.textContent = status + " " + percent.toFixed(1) + "%";
      outcome.title = count + " of " + rows.length + " cases";
      outcomes.appendChild(outcome);
    });

    const canvas = document.createElement("canvas");
    canvas.title = rows.map((row) => {
      const performance = row.status === "pass"
        ? " | " + Number(row.gflops).toPrecision(6) + " GFLOP/s"
        : "";
      return row.matrix_name + ": " + (row.status || "pass") + performance +
        " | preprocess " + Number(row.preprocess_ms).toPrecision(6) + " ms" +
        " | solve " + Number(row.runtime_ms).toPrecision(6) + " ms";
    }).join("\n");

    container.append(head, outcomes, canvas);
    plots.appendChild(container);
    requestAnimationFrame(() => drawPlot(canvas, rows));
  }
}

async function refreshResults() {
  const params = new URLSearchParams();
  selected("jobFilter").forEach((id) => params.append("job_id", id));
  selected("backendFilter").forEach((id) => params.append("backend_id", id));
  selected("suiteFilter").forEach((id) => params.append("suite", id));
  state.results = await getJson("/api/v1/results?" + params.toString());
  renderPlots(state.results);
}

async function refreshAll() {
  await refreshMetadata();
  await refreshResults();
}

window.addEventListener("pageshow", (event) => {
  if (event.persisted) window.location.reload();
});
byId("refresh").addEventListener("click", () => window.location.reload());
["jobFilter", "backendFilter", "suiteFilter"].forEach((id) =>
  byId(id).addEventListener("change", refreshResults)
);
refreshAll();
setInterval(refreshAll, 5000);

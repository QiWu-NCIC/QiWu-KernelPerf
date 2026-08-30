const state = {
  backends: [],
  suites: [],
  leaderboard: { enabled: false, public_url: "" },
  jobId: null,
};
const byId = (id) => document.getElementById(id);

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(url + ": " + response.status + " " + await response.text());
  return response.json();
}

function setStatus(message, error = false) {
  const node = byId("submitStatus");
  node.textContent = message;
  node.className = "submit-status" + (error ? " submit-error" : "");
}

function fillMetadata() {
  const backend = state.backends.find((item) => item.backend_id === byId("backend").value);
  const suite = state.suites.find((item) => item.suite_id === byId("suite").value);
  const operator = suite && suite.operators.find((item) => item.op_id === byId("operator").value);
  return { backend, operator };
}

function refreshBuildProfiles() {
  const backend = state.backends.find((item) => item.backend_id === byId("backend").value);
  const select = byId("buildProfile");
  const previous = select.value;
  select.replaceChildren(new Option("Default worker toolchain", ""));
  const profiles = (backend && backend.metadata && backend.metadata.build_profiles) || [];
  profiles.forEach((profile) => select.add(new Option(profile, profile)));
  select.value = profiles.includes(previous) ? previous : "";
}

function toggleArtifactFields() {
  const object = byId("artifactKind").value === "object";
  const sourceTree = byId("artifactKind").value === "source-tree";
  byId("sourceField").hidden = object || sourceTree;
  byId("sourceTextField").hidden = object || sourceTree;
  byId("sourceTreeFields").hidden = !sourceTree;
  byId("objectField").hidden = !object;
  byId("objectArchField").hidden = !object;
  if (object) {
    const backend = state.backends.find((item) => item.backend_id === byId("backend").value);
    byId("objectArch").value = (
      backend
      && backend.metadata
      && backend.metadata.labels
      && backend.metadata.labels.cuda_arch
    ) || "";
  }
}

async function fileBase64(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunkSize = 32768;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
}

async function sourceValue() {
  const file = byId("sourceFile").files[0];
  return file ? file.text() : byId("sourceText").value;
}

async function sourceTreeValue() {
  const selected = [...byId("sourceDirectory").files];
  if (!selected.length) throw new Error("Select a source directory.");
  if (selected.length > 512) throw new Error("Source directory exceeds 512 files.");
  const rawPaths = selected.map((file) => file.webkitRelativePath || file.name);
  const roots = new Set(rawPaths.map((value) => value.split("/", 1)[0]));
  const stripRoot = roots.size === 1 && rawPaths.every((value) => value.includes("/"));
  let totalBytes = 0;
  const files = [];
  for (let index = 0; index < selected.length; index += 1) {
    const file = selected[index];
    totalBytes += file.size;
    if (totalBytes > 8 * 1024 * 1024) throw new Error("Source directory exceeds 8 MiB.");
    const rawPath = rawPaths[index];
    const path = stripRoot ? rawPath.slice(rawPath.indexOf("/") + 1) : rawPath;
    files.push({ path, content: await file.text() });
  }
  return files;
}

async function submit(event) {
  event.preventDefault();
  const selected = fillMetadata();
  if (!selected.backend || !selected.operator) return;
  const kind = byId("artifactKind").value;
  const artifact = {
    name: byId("methodName").value.trim(),
    kind,
    metadata: {
      operator_id: selected.operator.op_id,
      base_format: byId("baseFormat").value.trim(),
    },
  };
  if (byId("configurationId").value.trim()) artifact.metadata.configuration_id = byId("configurationId").value.trim();
  if (byId("candidateGroup").value.trim()) artifact.metadata.candidate_group = byId("candidateGroup").value.trim();
  if (byId("buildProfile").value) artifact.metadata.build_profile = byId("buildProfile").value;
  if (kind === "source") {
    artifact.source = await sourceValue();
    if (!artifact.source.trim()) {
      setStatus("Provide source text or select a source file.", true);
      return;
    }
  } else if (kind === "source-tree") {
    try {
      artifact.kind = "source";
      artifact.source_files = await sourceTreeValue();
      artifact.entry_source = byId("entrySource").value.trim();
      artifact.compile_units = byId("compileUnits").value
        .split(/\r?\n/)
        .map((value) => value.trim())
        .filter(Boolean);
      if (!artifact.entry_source) {
        setStatus("Provide the adapter entry path.", true);
        return;
      }
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
  } else {
    const file = byId("objectFile").files[0];
    if (!file) {
      setStatus("Select an ELF relocatable .o file.", true);
      return;
    }
    artifact.object_base64 = await fileBase64(file);
    artifact.metadata.cuda_arch = byId("objectArch").value.trim();
  }
  setStatus("Submitting...");
  try {
    const response = await getJson("/api/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        generator_id: artifact.name,
        submitter: "kernelperf-web",
        backends: [selected.backend.backend_id],
        suites: [byId("suite").value],
        dataset_id: byId("dataset").value,
        operator_ids: [selected.operator.op_id],
        kernels: [artifact],
        tags: { base_format: artifact.metadata.base_format, artifact_kind: artifact.kind },
      }),
    });
    state.jobId = response.job_id;
    setStatus("Accepted " + response.job_id + ". Waiting for the worker...");
    await waitForJob(response.job_id, selected.backend.backend_id, byId("suite").value, selected.operator.op_id);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function waitForJob(jobId, backendId, suite, operatorId) {
  for (;;) {
    const job = await getJson("/api/v1/jobs/" + encodeURIComponent(jobId));
    setStatus(jobId + ": " + job.status);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      renderResult(job, backendId, suite, operatorId);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

function renderResult(job, backendId, suite, operatorId) {
  const node = byId("submitResult");
  node.hidden = false;
  node.innerHTML = "";
  const resultText = document.createElement("span");
  resultText.textContent = "Job " + job.status + ".";
  node.appendChild(resultText);
  if (job.status !== "succeeded") return;
  const download = document.createElement("a");
  download.className = "header-link";
  download.textContent = "Download result CSV";
  download.href = "/api/v1/results.csv?job_id=" + encodeURIComponent(job.job_id)
    + "&backend_id=" + encodeURIComponent(backendId)
    + "&suite=" + encodeURIComponent(suite) + "&operator_id=" + encodeURIComponent(operatorId)
    + (job.configuration_id ? "&configuration_id=" + encodeURIComponent(job.configuration_id) : "");
  download.download = "";
  const publish = document.createElement("button");
  publish.className = "primary-button";
  publish.textContent = "Submit to leaderboard";
  publish.disabled = !state.leaderboard.enabled;
  if (!state.leaderboard.enabled) {
    publish.title = "Leaderboard publishing is not configured on this server.";
  }
  publish.addEventListener("click", async () => {
    publish.disabled = true;
    publish.textContent = "Publishing...";
    try {
      const response = await getJson("/api/v1/leaderboard/submissions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: job.job_id, backend_id: backendId, suite, operator_id: operatorId }),
      });
      publish.textContent = "Published";
      if (response.leaderboard_url) window.open(response.leaderboard_url, "_blank", "noopener");
    } catch (error) {
      publish.disabled = false;
      publish.textContent = error.message;
    }
  });
  node.appendChild(download);
  node.appendChild(publish);
}

async function loadMetadata() {
  const [backends, suites, datasets, leaderboard] = await Promise.all([
    getJson("/api/v1/backends"),
    getJson("/api/v1/suites"),
    getJson("/api/v1/datasets"),
    getJson("/api/v1/leaderboard/config"),
  ]);
  state.backends = backends;
  state.suites = suites;
  state.leaderboard = leaderboard;
  backends.forEach((item) => byId("backend").add(new Option(item.name, item.backend_id)));
  suites.filter((item) => item.public).forEach((item) => byId("suite").add(new Option(item.display_name, item.suite_id)));
  byId("suite").value = suites.some((item) => item.suite_id === "spmv") ? "spmv" : (suites[0] && suites[0].suite_id);
  refreshSuiteFields(datasets);
  refreshBuildProfiles();
  if (leaderboard.public_url) byId("leaderboardLink").href = leaderboard.public_url;
  toggleArtifactFields();
}

function refreshSuiteFields(datasets) {
  const suite = state.suites.find((item) => item.suite_id === byId("suite").value);
  byId("operator").replaceChildren();
  if (suite) suite.operators.forEach((item) => byId("operator").add(new Option(item.name + " (" + item.dtype + ")", item.op_id)));
  byId("dataset").replaceChildren();
  datasets
    .filter((item) => item.dataset_id === (suite && suite.default_dataset_id) || item.dataset_id === "suitesparse_sample_100")
    .forEach((item) => byId("dataset").add(new Option(item.name || item.dataset_id, item.dataset_id)));
}

byId("artifactKind").addEventListener("change", toggleArtifactFields);
byId("backend").addEventListener("change", toggleArtifactFields);
byId("backend").addEventListener("change", refreshBuildProfiles);
byId("suite").addEventListener("change", async () => {
  const datasets = await getJson("/api/v1/datasets");
  refreshSuiteFields(datasets);
  refreshBuildProfiles();
  toggleArtifactFields();
});
byId("submitForm").addEventListener("submit", submit);
loadMetadata().catch((error) => setStatus(error.message, true));

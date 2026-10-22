export const cusparseCsrBestConfigurations = ["csr-default", "csr-alg1", "csr-alg2"];

export function deriveCusparseCsrBest(rows) {
  const candidates = rows.filter((row) => row.candidate_group === "cusparse"
    && cusparseCsrBestConfigurations.includes(row.configuration_id)
    && row.status === "pass" && Number(row.solve_ms) > 0);
  const versions = [...new Set(candidates.map((row) => (
    /^cuSPARSE CUDA ([^\s]+)/i.exec(row.method_name || "")?.[1]
  )).filter(Boolean))].sort();
  const methodName = versions.length
    ? `cuSPARSE CUDA ${versions.join("/")} CSR BEST`
    : "cuSPARSE CSR BEST";
  const byScope = new Map();
  for (const row of candidates) {
    const key = `${row.dtype}|${row.matrix_id}`;
    const previous = byScope.get(key);
    if (!previous || Number(row.solve_ms) < Number(previous.solve_ms)
      || (Number(row.solve_ms) === Number(previous.solve_ms)
        && row.configuration_id.localeCompare(previous.configuration_id) < 0)) {
      byScope.set(key, row);
    }
  }
  return [...byScope.values()].map((row) => ({
    ...row,
    method_id: "cusparse-csr-best",
    method_name: methodName,
    configuration_id: "per-matrix-best",
    candidate_group: "cusparse-csr-best",
    selection_role: "best",
    selected_from: cusparseCsrBestConfigurations.join(","),
    selected_configuration_id: row.configuration_id,
    base_format: "csr",
    public_ranked: true,
    ranking_scope: "main",
  }));
}

export function summarizeCusparseCsrBest(rows) {
  return ["fp32", "fp64"].map((dtype) => {
    const counts = new Map();
    for (const row of rows) {
      if (row.dtype !== dtype || row.status !== "pass") continue;
      const configuration = row.selected_configuration_id;
      if (configuration) counts.set(configuration, (counts.get(configuration) || 0) + 1);
    }
    const details = cusparseCsrBestConfigurations
      .filter((configuration) => counts.has(configuration))
      .map((configuration) => `${configuration.replace("csr-", "").toUpperCase()} ${counts.get(configuration)}`);
    return `${dtype.toUpperCase()}: ${details.length ? details.join(", ") : "no passing selections"}`;
  }).join(" | ");
}

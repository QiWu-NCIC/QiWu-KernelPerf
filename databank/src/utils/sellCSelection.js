function geometricMean(values) {
  const valid = values.filter((value) => Number.isFinite(value) && value > 0);
  return valid.length
    ? Math.exp(valid.reduce((sum, value) => sum + Math.log(value), 0) / valid.length)
    : 0;
}

export function sellCConfiguration(configurationId) {
  const match = /^sell-c(1|2|4|8|16|32|64|128)-(?:alg1|default)$/i.exec(
    String(configurationId || ""),
  );
  return match ? Number(match[1]) : null;
}

export function sellCWinnerByScope(
  source,
  efficiency,
  expectedCountForDataset,
  minimumCoverage = 1,
) {
  const groups = new Map();
  for (const row of source) {
    if (row.candidate_group !== "cusparse") continue;
    const c = sellCConfiguration(row.configuration_id);
    if (c === null || !["fp32", "fp64"].includes(row.dtype)) continue;
    const algorithm = row.configuration_id.endsWith("alg1") ? "alg1" : "default";
    const scope = `${row.backend_id}|${row.dataset_id || "none"}`;
    const key = `${scope}|${c}|${algorithm}|${row.dtype}`;
    if (!groups.has(key)) groups.set(key, { scope, c, algorithm, dtype: row.dtype, matrices: new Set(), scores: [] });
    const group = groups.get(key);
    if (row.matrix_id) group.matrices.add(row.matrix_id);
    const score = row.status === "pass" ? Number(efficiency(row)) : 0;
    if (score > 0) group.scores.push(score);
  }

  const candidates = new Map();
  for (const group of groups.values()) {
    const expected = Number(expectedCountForDataset(group.scope.split("|").at(-1), group.matrices.size));
    const coverage = expected > 0 ? group.matrices.size / expected : 0;
    const score = geometricMean(group.scores);
    if (coverage < minimumCoverage || score <= 0) continue;
    const key = `${group.scope}|${group.c}`;
    if (!candidates.has(key)) candidates.set(key, { scope: group.scope, c: group.c, scores: [] });
    candidates.get(key).scores.push(score);
  }

  const winners = new Map();
  for (const candidate of candidates.values()) {
    if (candidate.scores.length !== 4) continue;
    const score = geometricMean(candidate.scores);
    const previous = winners.get(candidate.scope);
    if (!previous || score > previous.score || (score === previous.score && candidate.c < previous.c)) {
      winners.set(candidate.scope, { c: candidate.c, score });
    }
  }
  return winners;
}

export function isPublicSpmvSellRow(row, winners) {
  if (row.candidate_group !== "cusparse") return true;
  const c = sellCConfiguration(row.configuration_id);
  if (c === null) return true;
  const scope = `${row.backend_id}|${row.dataset_id || "none"}`;
  const winner = winners.get(scope);
  return !winner || winner.c === c;
}

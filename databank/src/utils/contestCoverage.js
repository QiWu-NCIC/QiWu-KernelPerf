export function expectedMatrixCount(datasetId, observedCount, datasets) {
  const declaredCount = Number(datasets.find((dataset) => dataset.dataset_id === datasetId)?.matrix_count);
  return Number.isInteger(declaredCount) && declaredCount > 0 ? declaredCount : observedCount;
}

export function comparisonExclusions(datasetId, operator, dtype, datasets) {
  return datasets.find((dataset) => dataset.dataset_id === datasetId)
    ?.comparison_exclusions?.[operator]?.[dtype] || { matrix_ids: [], reason: "" };
}

export function isComparisonExcluded(row, datasets) {
  const operator = String(row.operator_id || "").split(".")[0];
  return comparisonExclusions(row.dataset_id, operator, row.dtype, datasets)
    .matrix_ids.includes(row.matrix_id);
}

export function comparisonMatrixCount(datasetId, operator, dtype, observedCount, datasets) {
  const excluded = new Set(comparisonExclusions(datasetId, operator, dtype, datasets).matrix_ids).size;
  return Math.max(0, expectedMatrixCount(datasetId, observedCount, datasets) - excluded);
}

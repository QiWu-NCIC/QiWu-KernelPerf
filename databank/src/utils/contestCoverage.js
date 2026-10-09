export function expectedMatrixCount(datasetId, observedCount, datasets) {
  const declaredCount = Number(datasets.find((dataset) => dataset.dataset_id === datasetId)?.matrix_count);
  return Number.isInteger(declaredCount) && declaredCount > 0 ? declaredCount : observedCount;
}

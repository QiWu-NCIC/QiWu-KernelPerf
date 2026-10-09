export function selectContestEntries(entries, { backend, dataset, rhs }) {
  return entries.filter((entry) => (!backend || entry.backend_id === backend)
    && (!dataset || entry.dataset_id === dataset)
    && (rhs === undefined || Number(entry.rhs_columns) === Number(rhs)));
}

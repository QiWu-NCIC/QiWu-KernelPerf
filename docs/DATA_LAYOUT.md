# Benchmark Data Layout

The static site treats `databank/public/data/index.json` as the authoritative
public manifest and `databank/public/data/candidate-pool/index.json` as the
authoritative candidate manifest. A CSV is never identified by a transient job
or submission UUID in its path.

## Active files

Results shown by the site are stored as:

```text
databank/public/data/results/<operator>/<backend>/<dataset>/
  <method_id>-<backend_id>-<dataset_id>-<dtype>.csv
```

The complete configuration sweep used for selection is stored as:

```text
databank/public/data/candidate-pool/<operator>/<backend>/<dataset>/
  <method_id>-<backend_id>-<dataset_id>-<dtype>.csv
```

The directory and filename both carry the dataset identity. `configuration_id`
remains in the manifest and CSV rows because it describes the candidate; method
IDs are required to be unique within a backend/dataset/dtype scope. This keeps
the stable public filename short while preventing a new dataset from replacing
an older candidate.

Each configuration in a candidate sweep must have a stable, distinct
`method_id` within its `operator/backend/dataset/dtype` scope (for example,
`cuSPARSE-CSR-ALG1` and `cuSPARSE-CSR-ALG2`). Do not reuse one method ID for
multiple configurations, and do not replace a dataset in place. A re-curated
matrix set gets a new immutable `dataset_id`; the migration and audit scripts
then keep both catalogs independent.

All curation, per-matrix `BEST`, coverage, and candidate-scope checks include
`operator_id`, `backend_id`, `dataset_id`, and `dtype`. A result with fewer than
90% passing matrix rows is retained in the candidate pool for diagnosis but is
not selected for a public aggregate.

## Migration and archives

Run the one-time layout migration from `databank` with Node.js 20 or newer:

```bash
npm run migrate:data-layout
npm run audit:spmv
```

The migration is collision-checked and preserves byte-identical files. Old
candidate CSVs that are not referenced by the manifest are removed; they are
not part of the published data contract. Keep any required historical copy
outside the repository.

`databank/public/issue_download` is not part of the current data contract. Its
legacy attachments are not tracked in this repository and have no reference
from the site. Keep them in external issue storage if provenance is required;
new issue attachments should enter through the normal review/import workflow
instead of becoming a second data source.

The evaluator's `KernelPerf/data/` directory is runtime state and is ignored by
Git. Its exports follow the same `<suite>/<backend>/<dataset>` partition before
they are manually imported into the versioned databank catalog.

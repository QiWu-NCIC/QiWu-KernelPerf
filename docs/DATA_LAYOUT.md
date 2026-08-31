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
candidate CSVs that were not referenced by the manifest are moved to
`public/data/archive/candidate-pool/unreferenced/`; they are not loaded by the
site. The archive is only for provenance and is not an input to curation.

`public/issue_download` is not part of the current data contract. Its legacy
attachments are retained under `docs/archive/issue_download` for provenance;
the site has no reference to them. New issue attachments should be handled by
the normal review/import workflow instead of being published as a second data
source.

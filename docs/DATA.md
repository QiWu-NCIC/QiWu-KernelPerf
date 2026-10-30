# Datasets And Results

`suitesparse_validation_100` is the legacy small-matrix corpus for correctness
and smoke checks. `suitesparse_sample_100` is the performance corpus: 100 real
SuiteSparse matrices, stratified over `100 <= nnz <= 100000000` and diversified
by row length and structural properties. Dataset IDs are immutable; a new
curation receives a new ID.

The Matrix Market reader handles real, integer, pattern, symmetric, Hermitian,
skew-symmetric, duplicate and deterministically sorted input. Complex matrices
are excluded from the performance corpus.

Regenerate and download the performance corpus from `KernelPerf/`:

```bash
python scripts/select_suitesparse.py \
  --output config/datasets/suitesparse_sample_100.json \
  --metadata-output config/datasets/suitesparse_sample_100.selection.json

python scripts/download_matrix_manifest.py \
  --manifest config/datasets/suitesparse_sample_100.json \
  --root /path/to/suitesparse
```

The JSON manifests are authoritative and the CSV files are committed with them
so GitHub Pages can serve downloads directly:

```text
databank/public/data/results/<operator>/<backend>/<dataset>/
  <method_id>-<backend_id>-<dataset_id>-<dtype>.csv
databank/public/data/candidate-pool/<operator>/<backend>/<dataset>/
  <method_id>-<backend_id>-<dataset_id>-<dtype>.csv
```

External result pull requests place unmodified evaluator exports temporarily
under `databank/result-submissions/spmv/`. CI validates those files without
adding them to the public index. An accepted file is imported into the layout
above and its inbox copy is removed before merge, so the published repository
does not retain duplicate CSV data.

Candidate sweeps are used for curation and per-matrix BEST. A method below 90%
successful coverage remains visible with failure counts but is not ranked.
Unreferenced legacy files and issue attachments are not tracked in the
repository; retain any provenance copy outside Git.

Each public index entry stores two distinct source fields. `source_sha256` is
the content hash of the source snapshot used for the recorded evaluation.
`source_manifest` is the downloadable, maintained plugin package generated
from the current canonical submission. When an implementation changes, keep
the evaluated hash unchanged; retain a separate submission directory if the
historical implementation must remain directly buildable.

## Failure diagnostics

SpMV and SpMM CSVs include `failure_reason`. Passing rows leave it empty;
failed and error rows contain a bounded diagnostic from validation metadata,
structured driver logs, compiler/runtime logs, or the reported exception.
Paths and credential-like values are redacted from the summary. Original
status, timings and measurements are not changed when enriching diagnostics.

For archived results, run the backfill utility from `KernelPerf/`:

```bash
python scripts/backfill_failure_reasons.py \
  --data-root ../databank/public/data \
  --database-root /path/to/archived-evaluator-databases \
  --report /path/outside/repository/failure-reasons.json
```

The default is a dry run; add `--write` to update CSVs and their existing index
hashes. Matching requires the same job, platform, operator, matrix, status,
timestamp and configuration; SpMM also requires the same RHS count and layout.
An older retry is not accepted as evidence for a newer failure. Missing or
conflicting historical evidence is explicitly marked `historical reason
unavailable`, never guessed. Existing diagnostics are preserved unless
`--refresh` is explicitly requested. The report lists unresolved files.
The utility verifies every original CSV field and row count before replacing
a file; it does not modify source snapshots or BEST selection.

Run `npm run audit:all` before publishing. The evaluator's
`KernelPerf/data/` directory is ignored runtime state; its exports use the same
suite/backend/dataset partition before manual import.

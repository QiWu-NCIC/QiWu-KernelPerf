# Release Review

## Scope

This review covers the unified `QiWu-KernelPerf` tree, the Python evaluator,
the reviewed SpMV submissions, and the static databank. The published CSV
files and JSON indexes are the release data; runtime databases, downloaded
matrix archives and build output are deliberately excluded.

## Integrated Changes

- Replaced five platform-specific remote regression scripts with the
  parameterized `KernelPerf/scripts/run_regression.py`. It discovers reviewed
  `submission.json` files, accepts an explicit submission/operator scope, keeps
  partial exports, and returns a non-zero status when any evaluation fails.
- Removed the obsolete online-issue importer, one-time migration/normalization
  tools, synthetic matrix generator, SQLite inspection helper, duplicate
  PlayGround artifact builder, and old standalone documentation.
- Fixed result export for jobs containing multiple candidate groups. Each
  `candidate_group` now receives an independent per-matrix BEST export.
  Offline BEST aggregation records every candidate configuration, including
  configurations that did not win a matrix.
- Made source generation deterministic from the canonical submissions. The
  ignored `public/source` tree is recreated before development, audits and
  production builds; it is not a second source tree or a release input.
- Simplified the databank to a Contest-first static page. Removed the unused
  router, portal view, Pinia, Element Plus, i18n, Axios, file-saver, Sass and
  related image assets. The production base path is `/QiWu-KernelPerf/`.
- Added the audit step to the Pages workflow and made submission CI run when
  evaluator, benchmark, configuration, or workflow code changes. Vite was
  updated to 6.4.3; `npm audit` reports no vulnerabilities.
- Moved all site-specific worker, scheduler, CUDA, GHOST, dataset and SSH
  settings into ignored `KernelPerf/config/private/service-*.json`,
  `workers-*.json` and `datasets-*.json` profiles. Tracked configuration files
  now use portable relative paths and a local-worker example.
- Kept the complete 100-row `sell-nrows` sweeps even when coverage is below the
  90% ranking threshold. Passing rows from every complete candidate sweep are
  included in per-matrix BEST selection; the threshold only controls ranking.
- Replaced the stale 5090 AlphaSparse exports with corrected,
  configuration-specific FP32 and FP64 runs and regenerated both BEST files.
- Removed obsolete remote/playground/debug scripts and unreferenced source
  packages from the release tree. The remaining scripts are reusable dataset,
  validation, regression, packaging, worker-installation and BEST tools.

## Verification

The following commands pass from a clean dependency installation:

```text
KernelPerf: python -m pytest -q       74 passed
databank:   npm run audit:spmv        364 public submissions, 0 failures
databank:   npm run build             Vite 6.4.3 production build succeeds
```

The audit reports 104 public entries for each of A100-SXM4-80GB,
H100-SXM5-80GB and RTX5090-SL3061, plus 26 entries for each of BW1000-gfx936
and Z100-gfx906. The candidate pool contains the complete configuration sweeps
used for BEST, including the 12 intentionally partial `sell-nrows` CSVs. No
private profile, SQLite runtime state, matrix archive, node dependency
directory or build directory belongs in the release package.

Public `source_sha256` values identify the evaluated source snapshot. The
download link resolves to the maintained standalone plugin generated from the
canonical submission, so a later source cleanup does not rewrite historical
evaluation provenance. This is intentional and is documented in `docs/DATA.md`.

## Remaining Release Decisions

1. Site-specific worker and dataset profiles now live in the ignored
   `KernelPerf/config/private/` directory. Keep that directory outside the
   published checkout; the tracked JSON files are portable examples. No private
   key or password is stored in the tree.
2. The repository has no top-level license, and the vendored AlphaSparse source
   does not declare a license in its upstream metadata. Redistribution and
   publication require an explicit license decision before release.
3. CUDA/GPU result identity is currently `method_id + backend_id + dataset_id +
   dtype`. If multiple CUDA versions for the same backend must coexist, the
   version must become part of the method identity (or a separate result field)
   in a future schema change.
4. The vendored AlphaSparse upstream tree is preserved as received and still
   contains upstream-language comments and documentation. All project-owned
   code, tests, release documentation and commit subjects are English. A
   literal no-Han-character policy for every vendored upstream file would
   require deleting or rewriting third-party source documentation.

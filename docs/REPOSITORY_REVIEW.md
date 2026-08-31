# Repository review

This is the maintenance boundary for the unified repository.

## Keep and extend

- `KernelPerf/kernelperf/`: evaluator core, scheduler, models, datasets and
  deterministic CSV export.
- `KernelPerf/benchmarks/`: one self-contained driver per operator.
- `KernelPerf/config/`: declarative worker, service, benchmark and dataset
  profiles. Platform-specific credentials stay outside Git.
- `KernelPerf/submissions/`: reviewed, standalone source packages and their
  upstream licenses.
- `KernelPerf/scripts/`: reproducible selection, download, validation,
  packaging, regression and BEST aggregation utilities.
- `databank/public/data/`: immutable result CSVs and generated indexes.
- `databank/public/source/`: generated plugin packages; never edit these files
  by hand.
- `databank/src/`: static Vue presentation only; ranking calculations remain
  client-side and data-driven.
- `.github/workflows/`: static checks and site deployment. Remote execution is
  intentionally deferred.

## Removed or superseded

- The old HTTP submission service, server-side Contest API and copied web UI are
  out of scope. Do not restore them in the unified repository.
- The former `kernelperf_spmv_*` and `abi_version` contracts are removed; use
  the standalone plugin header.
- Old UUID-based result filenames are historical data only. New exports use
  `method_id-backend_id-dataset_id-dtype.csv`.
- The former HTTP helper (`KernelPerf/scripts/common.py`) was unused after the
  evaluator became CLI-only and is deleted. The object-build helpers under
  `KernelPerf/scripts/playground/` are retained for now because they support
  the optional ELF submission path; they should be moved to a neutral
  `scripts/objects/` directory if object submissions become a primary mode.
- `KernelPerf/A100_PRE_GITHUB_TEST.md` was an environment-specific runbook for
  the former three-repository deployment and is deleted. Current procedures are
  in `docs/REGRESSION.md`.

## File hygiene

Do not commit `.venv`, `node_modules`, `dist`, `__pycache__`, SQLite databases,
remote logs, downloaded matrix archives, proxy settings or credentials. Keep
large upstream implementations only when their license and adapter make them
necessary for a standalone plugin. Temporary worker scripts belong on the
worker or in an issue, not in `KernelPerf/scripts`.

Before a release, run `git status --short`, `pytest -q`,
`python scripts/validate_submissions.py submissions`, `npm run audit:spmv`, and
the Node 20 production build. Any proposed deletion outside this list should
be checked against the source manifest, result index and reproducibility docs.

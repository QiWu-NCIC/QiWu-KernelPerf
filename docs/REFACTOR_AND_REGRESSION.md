# QiWu-KernelPerf refactor and regression report

## Release layout

`QiWu-KernelPerf` is now a standalone repository with two maintained areas:

- `KernelPerf/` contains the evaluator, benchmark drivers, worker and dataset
  configuration, and reviewed submissions under `submissions/<operator>/`.
- `databank/` contains the static leaderboard and incremental CSV/source data.

The old HTTP evaluator, Contest database, server-side submission endpoint, and
the copied KernelPerf Web UI were removed. Maintainers run
`python -m kernelperf.cli evaluate` against a reviewed checkout. GitHub Actions
only validates submission manifests, source limits, and Python tests.

## Dataset

`suitesparse_sample_100` is a deterministic real-only SuiteSparse selection with
100 unique matrices and nnz strata `10/14/18/20/20/18` for decimal orders
`10^2` through `10^7` (upper bound `10^8`). The legacy small corpus is now
`suitesparse_validation_100`. The selection is reproducible from the official
SuiteSparse statistics hash recorded in
`KernelPerf/config/datasets/suitesparse_sample_100.selection.json`.

## Local verification

From `KernelPerf/`:

```text
python -m pytest -q                 # 46 passed
python scripts/validate_submissions.py submissions
python -m compileall -q kernelperf benchmarks scripts
```

The databank audit reports 156 retained historical submissions (52 per backend)
and no missing source or metadata failures. The static site builds with Node 20
using `npx --yes node@20 node_modules/vite/bin/vite.js build`.

## Remote verification

- H100 login (`qiwu-h100`) works with the configured ECDSA key. GPU commands must
  run inside `srun --account=ncic --gres=gpu:1`; a one-matrix cuSPARSE FP32
  smoke completed and produced 27 candidate CSV files including BEST.
- H100 and RTX 5090 matrix downloads use the retrying downloader and the new
  manifest. H100 reached 96/100 before the four largest archives timed out in
  the proxy; the downloader is resumable. RTX 5090 reached 80/100 when this
  report was written.
- A100 dual-factor SSH was verified (`SXMa100`, eight A100 GPUs). The gateway
  permits authentication and directory creation but did not provide a stable
  non-interactive shell for the long download command; no A100 performance
  result is claimed from this run.

The H100 full regression is launched with
`scripts/remote_h100_full_regression.sh`; it preserves partial CSV exports and
records failed or missing matrices rather than fabricating successful cases.

## Known operational constraints

The four largest SuiteSparse archives may require a direct network path or a
long-lived proxy connection. Before publishing results, rerun the downloader
until all 100 files exist, then copy the generated CSVs into `databank/public/data`
and run `npm run audit:spmv`.

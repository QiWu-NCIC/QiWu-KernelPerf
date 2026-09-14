# Release package

This tree is ready to publish as `QiWu-KernelPerf`. The release keeps the
evaluator, reviewed SpMV submissions, static leaderboard source, and the
versioned result catalog together.

## Included

- `KernelPerf/`: evaluator, benchmark drivers, portable configuration examples,
  tests, reusable scripts, and reviewed submissions.
- `databank/public/data/`: JSON manifests and CSV results for A100-SXM4-80GB,
  H100-SXM5-80GB and RTX5090-SL3061. Both `suitesparse_sample_100` and
  `suitesparse_validation_100` are retained.
- Source snapshots are generated into `databank/public/source/` from
  `KernelPerf/submissions/spmv` by `npm run prepare:sources`. This directory is
  ignored and is included only in the generated Pages artifact, not in Git.
- `docs/`: contribution, data, operations and release instructions.

The SpMV correctness protocol uses 5 warmups, 20 timed solves, and the dynamic
row bound `abs(y-y_ref)/s_i <= 4*n_i*u`, with `u` taken from the tested `float`
or `double` storage. Both dtypes use a host `long double` reference; products
are promoted to `long double` before accumulation. The authoritative configuration and implementation are
documented in the root README and `KernelPerf/benchmarks/spmv/template.cu`.

The candidate pool intentionally retains complete 100-row sweeps with failed
cases, including the 12 `sell-nrows` files. Passing rows from those sweeps are
valid inputs to per-matrix BEST; the 90% coverage rule only controls ranking.

## Excluded

Do not publish site credentials or machine-specific settings from
`KernelPerf/config/private/`. Runtime SQLite files, downloaded matrices,
`node_modules`, `dist`, `public/source`, Python caches, logs, archives and
temporary exports are also excluded. The tracked `config/*.json` files are
portable examples.

## Verification

From a clean checkout:

```bash
cd KernelPerf
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/validate_submissions.py submissions

cd ../databank
npm ci
npm run audit:spmv
npm run build
```

The release audit should report `312 public submissions` and `failures: []`.
The generated Pages site is `databank/dist`; it is deployment output and is
not committed to the repository. `databank/public/source` is a generated build
input and must not be committed.

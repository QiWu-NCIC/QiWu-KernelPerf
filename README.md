# QiWu-KernelPerf

QiWu-KernelPerf is a maintainer-run benchmark repository for GPU kernels. It
contains the Python evaluator, reviewed submissions, reproducible datasets,
and the static result databank in one repository.

**Leaderboard:** [QiWu-KernelPerf Contest](https://qiwu-ncic.github.io/QiWu-KernelPerf/)

## Repository layout

```text
QiWu-KernelPerf/
|-- KernelPerf/
|   |-- kernelperf/          evaluator, scheduler, models, CSV export
|   |-- benchmarks/          operator drivers (SpMV is the reference driver)
|   |-- config/              portable examples; private site profiles stay ignored
|   |-- submissions/         reviewed plugins: <operator>/<method>/
|   |-- scripts/             dataset, packaging, validation and regression tools
|   `-- tests/               evaluator and plugin contract tests
|-- databank/
|   |-- public/data/         immutable CSV results and JSON indexes
|   |-- public/source/       generated standalone source packages (ignored)
|   |-- src/                 Vue static leaderboard
|   `-- scripts/             source generation, result import and data audits
|-- docs/                    maintained operating and design documentation
`-- .github/workflows/       PR validation and static-site deployment
```

There is no public server-side submission API. A contributor opens a pull
request that adds a plugin under `KernelPerf/submissions/<operator>/<method>/`.
GitHub Actions checks the manifest, paths, source size and Python tests. After
review, a maintainer checks out the PR on an isolated worker, runs the evaluator,
reviews the CSVs, and updates `databank/public/data` in a separate change.

## Quick start

### Evaluator

```bash
cd KernelPerf
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_submissions.py submissions
```

Evaluate one reviewed submission on a configured worker:

```bash
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --submission submissions/spmv/cusparse \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32
```

The tracked configuration is a portable local example. Maintainer GPU runs use
an ignored site profile, for example
`config/private/service-h100.json`; see
[`KernelPerf/config/README.md`](KernelPerf/config/README.md).

The evaluator owns matrix loading, correctness checks, warmup/repeat timing and
CSV export. Result files are written below
`KernelPerf/data/result_exports/<suite>/<backend>/<dataset>/` with the canonical
name `method_id-backend_id-dataset_id-dtype.csv`.

Use `KernelPerf/config/private/service-<platform>.json` for maintainer GPU
profiles; those files are intentionally excluded from the published tree.

### SpMV measurement protocol

The published SpMV settings come from
[`KernelPerf/config/benchmarks.json`](KernelPerf/config/benchmarks.json). The
current FP32 and FP64 profiles both use 5 untimed warmup calls followed by 20
timed solve calls for each matrix. The standard CUDA path records one CUDA
Event interval around the complete repeat loop and reports the interval divided
by 20. A plugin that defines `KERNELPERF_SPMV_HOST_TIMING` instead uses a host
steady clock and a final device synchronization. Warmup, preprocessing,
validation and teardown are excluded from `solve_ms`.

`preprocess_ms` is host wall time around `qiwu_spmv_preprocess`, ending after a
stream synchronization. It therefore includes CPU work, allocations and data
movement performed by the plugin. The evaluator's initial CSR upload happens
before this interval. The leaderboard derives its other time modes as
`preprocess_ms + solve_ms` and `preprocess_ms / iteration + solve_ms`.

Correctness is checked after timing with one separate solve call. The reference
is CSR SpMV on the CPU: values and the input vector use the tested storage type,
while both FP32 and FP64 rows are accumulated in host `long double`. Let `n_i` be the row's
nonzero count, `s_i = sum_j(abs(a_ij*x_j))`, and `u = eps(storage_type) / 2`.
A finite result passes when every non-noise row satisfies

```text
abs(y - y_ref) / s_i <= C * n_i * u
```

The safety factor is `C=4`. `u` is computed from the actual output storage:
`2^-24` for FP32 (`float`) and `2^-53` for FP64 (`double`). Rows whose
`(s_i / abs(y_ref_i)) * n_i * u >= 1` are reported as numerical-noise rows and
excluded from the error aggregate; zero-scale rows still require an exactly
zero output. Reported absolute and ordinary relative errors are diagnostics;
the dynamic row bound above is the pass/fail criterion. The lifecycle and timing implementation is in
[`KernelPerf/benchmarks/spmv/template.cu`](KernelPerf/benchmarks/spmv/template.cu),
and the configuration handoff and result parsing are in
[`KernelPerf/benchmarks/spmv/driver.py`](KernelPerf/benchmarks/spmv/driver.py).

### Static databank

```bash
cd databank
npm ci                         # Node.js 20 or newer
npm run audit:spmv
npm run dev
```

Open the Vite URL printed by the command. The browser computes GFLOP/s,
solve-only efficiency, geometric means and coverage from the CSV data. A
method with at least 90% successful cases for both FP32 and FP64 is rankable;
partial methods remain visible with their failed-case count.

The published catalog uses the same identity in its directory layout:
`databank/public/data/results/<operator>/<backend>/<dataset>/` for ranked
results and `databank/public/data/candidate-pool/<operator>/<backend>/<dataset>/`
for complete configuration sweeps. The JSON manifests are authoritative; see
[`docs/DATA.md`](docs/DATA.md).

## Current scope

The reference operator is CSR-input SpMV with FP32 and FP64 variants. The
baseline plugins cover cuSPARSE format/configuration sweeps, CSR5,
CSR-Adaptive, AlphaSparseLib, rocSPARSE and GHOST SELL-C-sigma. The BW1000 HIP
regression scope uses only the native HIP AlphaSparseLib and rocSPARSE
submissions. New operators should add
an independent driver under `KernelPerf/benchmarks/<operator>/` and declare it
in `KernelPerf/config/benchmarks.json`; the scheduler does not need operator-
specific changes.

The performance corpus is `suitesparse_sample_100`, a deterministic set of 100
real SuiteSparse matrices with `10^2 <= nnz <= 10^8`. The smaller
`suitesparse_validation_100` corpus is retained for correctness and smoke
checks. Selection and download instructions are in
[`docs/DATA.md`](docs/DATA.md).

## Contribution and release flow

1. Add or update one self-contained plugin and its `submission.json`.
2. Run the local validation and tests, then open a pull request.
3. A maintainer evaluates the reviewed commit on each target platform.
4. Copy the generated CSVs and metadata into `databank/public/data`.
5. Run the databank audit and build; source packages are generated locally and
   included in the Pages artifact, but are not committed.

## Release package

The hand-off package contains the tracked evaluator, reviewed submissions,
portable configuration examples, documentation, and the versioned databank
under `databank/public/data`. It excludes site-private profiles, runtime
SQLite files, downloaded matrix archives, dependency directories, build output
and `databank/public/source`. Source packages are generated from the reviewed
submissions by `npm run prepare:sources` before the static site is built.

After unpacking a release package, install dependencies and verify it with:

```bash
cd KernelPerf
python -m pip install -e '.[dev]'
python -m pytest -q

cd ../databank
npm ci
npm run audit:spmv
npm run build
```

Plugins expose the source-level `qiwu_spmv_preprocess`,
`qiwu_spmv_solve` and `qiwu_spmv_destroy` functions. The downloaded package
contains its own public header, adapter and standalone CMake example, so it can
be built without importing the Python evaluator. See
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Documentation map

- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md): plugin contract and pull requests.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md): maintainer workflow and data ownership.
- [`docs/DATA.md`](docs/DATA.md): datasets, CSV data and download layout.
- [`docs/FUTURE_REMOTE_ACTIONS.md`](docs/FUTURE_REMOTE_ACTIONS.md): explicitly deferred protected-runner design.
- [`docs/RELEASE_REVIEW.md`](docs/RELEASE_REVIEW.md): release cleanup, verification and remaining decisions.
- [`docs/RELEASE_PACKAGE.md`](docs/RELEASE_PACKAGE.md): publishable tree, data ownership and hand-off checks.

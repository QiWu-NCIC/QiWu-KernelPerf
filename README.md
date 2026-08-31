# QiWu-KernelPerf

QiWu-KernelPerf is a maintainer-run benchmark repository for GPU kernels. It
contains the Python evaluator, reviewed submissions, reproducible datasets,
and the static result databank in one repository.

## Repository layout

```text
QiWu-KernelPerf/
|-- KernelPerf/
|   |-- kernelperf/          evaluator, scheduler, models, CSV export
|   |-- benchmarks/          operator drivers (SpMV is the reference driver)
|   |-- config/              workers, datasets, benchmarks, service profiles
|   |-- submissions/         reviewed plugins: <operator>/<method>/
|   |-- scripts/             dataset, packaging, validation and regression tools
|   `-- tests/               evaluator and plugin contract tests
|-- databank/
|   |-- public/data/         immutable CSV results and JSON indexes
|   |-- public/source/       generated standalone source packages
|   |-- src/                 Vue static leaderboard
|   `-- scripts/             source generation, curation and data audits
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

The evaluator owns matrix loading, correctness checks, warmup/repeat timing and
CSV export. Result files are written below
`KernelPerf/data/result_exports/<suite>/<backend>/<dataset>/` with the canonical
name `method_id-backend_id-dataset_id-dtype.csv`.

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
CSR-Adaptive, AlphaSparseLib and GHOST SELL-C-sigma. New operators should add
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
4. Copy the generated CSVs, source package and metadata into `databank`.
5. Run the databank audit and build; deploy the static site only after review.

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

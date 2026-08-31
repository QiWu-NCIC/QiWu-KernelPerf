# Operations Guide

## Repository responsibilities

- `KernelPerf/kernelperf/` is the evaluator runtime and scheduler. It has no
  public HTTP submission endpoint.
- `KernelPerf/benchmarks/` contains operator drivers and templates.
- `KernelPerf/submissions/<operator>/<method>/` contains reviewed adapters and
  source/object artifacts. `submission.json` is the only evaluator manifest.
- `KernelPerf/config/` contains checked-in benchmark, dataset, and worker
  profiles. Machine-specific credentials and temporary fallback profiles stay
  outside the repository.
- `KernelPerf/data/` is runtime state. SQLite files and result exports are
  generated outputs and are ignored by git.
- `databank/public/data/` is the publishable CSV catalog. `databank/public/source/`
  is generated from the canonical submissions and should not be edited by hand.

## Maintainer workflow

1. Review a pull request and run the GitHub Actions validation workflow.
2. On an isolated worker, run `python -m kernelperf.cli evaluate` for each
   requested operator and dtype. Keep the checked-out PR source read-only.
3. Copy the generated CSV files from `KernelPerf/data/result_exports/` into the
   matching `databank/public/data/results/<operator>/<backend>/` directory.
4. Run `npm run audit:spmv` in `databank`, then build the static site. Commit the
   CSV files, index metadata, and generated source package together.

The evaluator writes one file per method, backend, dataset, and dtype. A
transient job ID remains in each row for traceability but is deliberately absent
from the final filename.

## Remote profiles

The checked-in worker profiles describe the normal CUDA workers. If a login
node must be used temporarily, create an untracked service/worker profile with
an explicit backend ID, CUDA version, dataset root, and result directory. Do
not replace the checked-in profile or publish fallback measurements without
recording the actual CUDA/toolkit version in the result metadata.

Large SuiteSparse archives should be downloaded once into the configured root
and reused by all baseline groups. The downloader is resumable; verify all 100
manifest names before starting a full regression.


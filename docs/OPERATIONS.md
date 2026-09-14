# Maintainer Operations

`KernelPerf/kernelperf` is the evaluator and scheduler. `KernelPerf/submissions`
contains reviewed plugins. `KernelPerf/data` is ignored runtime state.
`databank/public/data` is the versioned CSV catalog. `databank/public/source`
is an ignored build directory generated from canonical submissions.

## Evaluate and publish

From `KernelPerf/`, evaluate every reviewed submission for one platform and
dataset with the parameterized regression runner:

```bash
python scripts/run_regression.py \
  --config config/private/service-a100.json \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100
```

Use `--submission` and repeated `--operator` options to narrow a run. The
portable `config/service.json` is only a local example; platform profiles are
kept under the ignored `config/private/` directory. Isolate
long sweeps with independent SQLite and export directories; preserve rows for
failed matrices. Exports are written below
`KernelPerf/data/result_exports/<suite>/<backend>/<dataset>/`.

Copy accepted results into the matching databank scope. Use
`npm run add:spmv -- <file.csv>` for a single public result, or
`--candidate` for a configuration sweep. Then run:

```bash
cd databank
npm run audit:spmv
npm run build
```

Commit CSV files and JSON indexes together. Before local development, audits or
deployment, the databank scripts generate standalone source packages from
`KernelPerf/submissions`; the browser then performs GFLOP/s, efficiency,
geometric-mean, coverage and BEST calculation from the GitHub-hosted CSV files.

## Platform notes

Use the private worker profiles under `KernelPerf/config/private/` and record
the exact worker label, CUDA toolkit or HIP runtime version, dataset manifest
hash, commit SHA, configuration IDs and pass/fail
counts. Credentials, temporary fallback profiles and downloaded matrix archives
stay outside Git. Use a Slurm GPU allocation on platforms whose login node has
no GPU, and give parallel batches independent SQLite and export paths.

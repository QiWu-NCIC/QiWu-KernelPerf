# Platform regression

Regression is a maintainer operation performed from a reviewed checkout. It is
not part of the GitHub Actions job and it never accepts arbitrary network
submissions.

## Procedure

From `KernelPerf/`, use the platform-specific service and worker profile:

```bash
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --submission submissions/spmv/cusparse \
  --operator spmv.csr.fp32
```

Repeat for FP64 and each reviewed baseline. Long sweeps should run one
configuration at a time (or in isolated copies with independent SQLite and
export directories). Preserve partial CSVs when a matrix fails; a failed case
must be represented by a row instead of being silently removed.

## Results

The evaluator writes CSVs under
`KernelPerf/data/result_exports/<suite>/<backend>/<dataset>/`. The filename contains
only method, backend, dataset and dtype. The CSV keeps the transient job ID for
traceability. Before publishing, generate per-matrix BEST files from passing
rows only, copy the accepted files to
`databank/public/data/results/<operator>/<backend>/<dataset>/`, update the JSON
index, and run:

```bash
cd databank
npm run audit:spmv
npx --yes node@20 node_modules/vite/bin/vite.js build
```

The leaderboard ranks a method only when both dtypes meet the 90% successful
coverage threshold. It still displays lower-coverage methods and their failed
case counts.

## Platform notes

- `A100-SXM4-80GB`: CUDA 12.2 is available on the current login node; the
  original CUDA 12.9 second-hop worker requires separate credentials.
- `RTX5090-SL3061`: use the CUDA 12.8 profile and isolate each GPU's SQLite and
  result directory when running parallel batches.
- `H100-SXM5-80GB`: login nodes do not expose GPUs; execute the evaluator inside
  a Slurm GPU allocation with the CUDA 12.8 profile.

Record the exact worker label, CUDA version, dataset manifest hash, commit SHA,
configuration IDs and pass/fail counts with every release update.

# Maintainer Operations

`KernelPerf/kernelperf` is the evaluator and scheduler. `KernelPerf/submissions`
contains reviewed plugins. `KernelPerf/data` is ignored runtime state.
`databank/public/data` is the versioned CSV catalog and `public/source` is
generated from canonical submissions.

## Evaluate and publish

From `KernelPerf/`, evaluate each requested dtype and platform:

```bash
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --submission submissions/spmv/my-method \
  --operator spmv.csr.fp32
```

Repeat for FP64 and each reviewed baseline. Isolate long sweeps with independent
SQLite and export directories; preserve rows for failed matrices. Exports are
written below `KernelPerf/data/result_exports/<suite>/<backend>/<dataset>/`.

Copy accepted results into the matching databank scope. Use
`npm run add:spmv -- <file.csv>` for a single public result, or
`--candidate` for a configuration sweep. Then run:

```bash
cd databank
npm run curate:spmv
npm run audit:spmv
npm run build
```

Commit CSV files, JSON indexes, and generated source packages together. The
browser performs GFLOP/s, efficiency, geometric-mean, coverage and BEST
calculation from these GitHub-hosted files.

## Platform notes

Use the checked-in worker profiles and record the exact worker label, CUDA
version, dataset manifest hash, commit SHA, configuration IDs and pass/fail
counts. Credentials, temporary fallback profiles and downloaded matrix archives
stay outside Git. Use a Slurm GPU allocation on platforms whose login node has
no GPU, and give parallel batches independent SQLite and export paths.

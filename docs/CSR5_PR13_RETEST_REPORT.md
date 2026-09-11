# CSR5 PR #13 Retest Report

## Code version

CSR5 uses the fixed version from upstream [pull request #13](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/pull/13),
pinned at commit `caff9d80665052bd5379fb37fd135f0d72d7b525`. On top of the original CUDA fix, this version adds an
optional `cudaStream_t` to `anonymouslibHandle::spmv` and makes the CSR5 compute, calibration and tail kernels run on
the caller's stream. The QiWu adapter only copies the CSR input (CSR5 converts column indices and values in place),
manages lifetimes and adapts the interface; it does not modify the CSR5 algorithm itself.

## Test methodology

- Dataset: `suitesparse_sample_100`, 100 real-valued SuiteSparse matrices.
- Operator: CSR SpMV, with FP32 and FP64 tested separately.
- Per matrix: 5 warm-up iterations and 20 timed iterations measured with CUDA events; preprocessing time is recorded separately.
- Correctness: a high-precision CPU CSR reference is used, with a dynamic per-row error bound based on the actual output storage precision.
- Results: 100 FP32 plus 100 FP64 cases per platform, 200 matrix cases in total.

## Platform results

| Platform | FP32 | FP64 | Result files |
| --- | ---: | ---: | --- |
| A100-SXM4-80GB | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/A100-SXM4-80GB/suitesparse_sample_100/` |
| H100-SXM5-80GB | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/H100-SXM5-80GB/suitesparse_sample_100/` |
| RTX5090-SL3061 | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/RTX5090-SL3061/suitesparse_sample_100/` |

All six CSV files contain 100 rows and 100 unique matrices with no failure records. The result index and the candidate
pool have been updated to reference the current CSR5 plugin content hash and the PR #13 version information.

## H100 environment notes

One retest on the H100 node hit a 1200-second timeout on a single matrix. Diagnosis showed that the node was
simultaneously running model serving / MPS, so GPU memory and compute resources were occupied by external processes;
the same binary completed a small-matrix diagnostic on an idle node and eventually finished the full 200/200 run.
Therefore the published data only uses complete runs without external GPU processes, and timed-out batches are discarded.

## Verification commands

```bash
cd KernelPerf
python -m pytest -q
python scripts/validate_submissions.py submissions

cd ../databank
npm run audit:spmv
npm run build
```

This audit reports `312 public submissions`, `104` entries per backend, and `failures: []`.

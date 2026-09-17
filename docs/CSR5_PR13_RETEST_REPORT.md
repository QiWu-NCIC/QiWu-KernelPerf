# CSR5 PR #13 Retest Report

## Code version

The CSR5 submission uses the corrected upstream pull request
[#13](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/pull/13),
pinned at commit `caff9d80665052bd5379fb37fd135f0d72d7b525`. The patch adds an
optional `cudaStream_t` to `anonymouslibHandle::spmv` and routes the CSR5
compute, calibration, and tail kernels through the caller's stream. The QiWu
adapter only copies the CSR input, manages lifetime, and connects the plugin
interface; it does not change the CSR5 algorithm.

## Test protocol

- Dataset: `suitesparse_sample_100`, 100 real SuiteSparse matrices.
- Operator: CSR SpMV, tested independently in FP32 and FP64.
- Per matrix: 5 warmups and 20 CUDA-event timed solve iterations; preprocessing
  time is recorded separately.
- Correctness: a host high-precision reference with the dynamic row error
  bound documented in the root README.
- Scope: 100 FP32 and 100 FP64 cases per platform.

## Platform results

| Platform | FP32 | FP64 | Result directory |
| --- | ---: | ---: | --- |
| A100-SXM4-80GB | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/A100-SXM4-80GB/suitesparse_sample_100/` |
| H100-SXM5-80GB | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/H100-SXM5-80GB/suitesparse_sample_100/` |
| RTX5090-SL3061 | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/RTX5090-SL3061/suitesparse_sample_100/` |

Each result CSV contains 100 unique matrices and no failed cases. The result
indexes and candidate pool reference the current CSR5 source package hash and
the pinned PR version.

## H100 environment note

An intermediate H100 run hit a 1200-second timeout while model-serving/MPS
processes occupied the GPU. The final published data comes from a clean node
and contains all 200 cases; the interrupted batch is not retained.

## Verification commands

```bash
cd KernelPerf
python -m pytest -q
python scripts/validate_submissions.py submissions

cd ../databank
npm run audit:spmv
npm run build
```

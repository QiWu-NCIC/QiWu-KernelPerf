# CUDA CSR-Adaptive submission

This submission is a CUDA port of the CSR-Adaptive policy from the upstream
clSPARSE `csrmv_adaptive.cl` kernel. The OpenCL backend is not required: the
single `adapter.cu` file implements the standard KernelPerf lifecycle and
dispatches short rows through scalar threads and longer rows through one warp.

The fixed policy constants follow the upstream configuration (`WG_SIZE=256`,
`BLKSIZE=1024`, `BLOCK_MULTIPLIER=3`, `ROWS_FOR_VECTOR=1`, `ROW_BITS=32`, and
`WG_BITS=24`). `preprocess` builds the dynamic row-block grid used by the
original `rowBlocks` path and uploads its compact plan. CSR input remains in
the evaluator-owned device buffers.

Example FP32 submission:

```bash
python scripts/submit_spmv.py \
  --api http://127.0.0.1:18081 \
  --backend A100-SXM4-80GB \
  --operator spmv.csr.fp32 \
  --method-name csr-adaptive-cuda \
  --base-format csr \
  --source-dir examples/csr_adaptive_submission \
  --entry-source adapter.cu \
  --wait
```

Use `spmv.csr.fp64` for the FP64 candidate.

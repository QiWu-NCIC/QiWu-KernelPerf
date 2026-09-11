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
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --backend A100-SXM4-80GB \
  --operator spmv.csr.fp32 \
  --submission submissions/spmv/csr_adaptive
```

Use `spmv.csr.fp64` for the FP64 candidate.

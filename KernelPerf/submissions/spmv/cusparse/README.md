# cuSPARSE format and algorithm submissions

This source tree adapts the official cuSPARSE Generic API supplied by the
NVIDIA CUDA Toolkit. It is intended to be built and evaluated with the same
CUDA 12.8 toolkit on every backend (12.8.2 on H100 and 12.8.93 on RTX 5090).

The shared implementation is [`adapter.cu`](adapter.cu).
Each file under `variants/` is an entry source; select exactly one as
`entry_source` when submitting the directory. The entry source only selects a
format, algorithm, and (for SELL) slice size. All conversion and lifecycle
logic remains in the shared adapter.

Supported entries:

- `csr_default.cu`, `csr_alg1.cu`, `csr_alg2.cu`
- `coo_default.cu`, `coo_alg1.cu`, `coo_alg2.cu`
- `csc_default.cu`, `csc_alg1.cu`, `csc_alg2.cu`
- `sell_c{1,2,4,8,16,32,64,128}_{default,alg1}.cu`
- `sell_nrows_default.cu`, `sell_nrows_alg1.cu`

The `sell_nrows` entries intentionally use cuSPARSE Sliced-ELL with
`sliceSize = nrows`, producing one full-height slice. This is the compatible
SpMV descriptor path; Blocked-ELL is an SpMM format and is not used here.

Use `cusparse_sell_nrows` as the method name when submitting either of these
entries; keep `base_format` set to `sell`.

For example, the ALG1 candidate is submitted with:

```bash
python scripts/submit_spmv.py \
  --api http://127.0.0.1:18081 \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32 \
  --method-name cusparse_sell_nrows \
  --base-format sell \
  --source-dir examples/cusparse_submission \
  --entry-source variants/sell_nrows_alg1.cu \
  --wait
```

Example FP32 submission for CSR algorithm 2:

```bash
python scripts/submit_spmv.py \
  --api http://127.0.0.1:18081 \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32 \
  --method-name cuSPARSE-CSR-ALG2 \
  --base-format csr \
  --source-dir examples/cusparse_submission \
  --entry-source variants/csr_alg2.cu \
  --wait
```

Replace the operator with `spmv.csr.fp64` for FP64. `preprocess` creates the
format descriptor, uploads converted storage, allocates the cuSPARSE external
buffer, and calls `cusparseSpMV_preprocess`; `solve` only calls
`cusparseSpMV` on the evaluator-provided stream.

CUDA 12.8 exposes `CUSPARSE_SPMV_ALG_DEFAULT`, COO ALG1/ALG2, CSR ALG1/ALG2,
and SELL ALG1. The CSR algorithms are also the descriptor algorithms for CSC.
The two SELL entries per slice size are therefore the default algorithm and
`CUSPARSE_SPMV_SELL_ALG1`.

After all candidate jobs finish, derive the per-matrix `CUSPARSE_BEST` result
without mixing preprocessing into solve timing:

```bash
python scripts/select_cusparse_best.py \
  --input data/result_exports/spmv/A100-SXM4-80GB/cusparse-csr.csv \
  --input data/result_exports/spmv/A100-SXM4-80GB/cusparse-coo.csv \
  --input data/result_exports/spmv/A100-SXM4-80GB/cusparse-sell-c32.csv \
  --output data/result_exports/spmv/A100-SXM4-80GB/cusparse-best.csv
```

The selector keeps the fastest passing row for each matrix and marks the
derived method as `CUSPARSE_BEST` with `base_format=auto`.

For a single sweep job, use the managed per-matrix-best flow instead:

```bash
python scripts/submit_spmv.py \
  --api http://127.0.0.1:18081 \
  --backend A100-SXM4-80GB \
  --operator spmv.csr.fp32 \
  --base-format auto \
  --candidate-group cusparse-csr \
  --source-dir examples/cusparse_submission \
  --sweep-manifest examples/cusparse_submission/sweep_manifest.json \
  --wait --publish
```

The service exports every candidate configuration and an additional
`configuration_id=per-matrix-best` CSV. BEST selects the lowest passing
`solve_ms` independently for every matrix.

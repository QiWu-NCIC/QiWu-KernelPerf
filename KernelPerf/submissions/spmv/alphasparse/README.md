# AlphaSparseLib CSR SpMV adapter

These candidates port the CSR algorithm families exposed by
`alphasparse_for_test/hip/kernel/level2/alphasparse_spmv.hip` to the CUDA
lifecycle contract used by KernelPerf. The submitted tree contains only the
adapter, selectors, and provenance metadata; the complete AlphaSparseLib source
tree is not uploaded for each job.

All seven candidates use the same `adapter.cu` lifecycle implementation; each
entry source only selects one upstream algorithm family:

`scalar`, `vector`, `merge`, `line-enhance`, `flat1`, `flat4`, and `flat8`.

Example:

```bash
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --backend A100-SXM4-80GB \
  --operator spmv.csr.fp32 \
  --submission submissions/spmv/alphasparse \
  --configuration-id vector
```

The adapter keeps CSR data in the framework-owned device buffers. Merge uses a
two-stage product/reduction workspace allocated in `preprocess`, while the
other methods need no extra storage. FP64 uses the same kernels and is selected
by the normal `spmv.csr.fp64` operator.

To evaluate all seven configurations in one job, create a manifest such as:

```json
[
  {"configuration_id":"scalar", "entry_source":"variants/scalar.cu"},
  {"configuration_id":"vector", "entry_source":"variants/vector.cu"},
  {"configuration_id":"merge", "entry_source":"variants/merge.cu"},
  {"configuration_id":"line-enhance", "entry_source":"variants/line_enhance.cu"},
  {"configuration_id":"flat1", "entry_source":"variants/flat1.cu"},
  {"configuration_id":"flat4", "entry_source":"variants/flat4.cu"},
  {"configuration_id":"flat8", "entry_source":"variants/flat8.cu"}
]
```

Evaluate each configuration with `--configuration-id`; the maintainer can then
run `scripts/aggregate_best.py` over the resulting candidate CSVs.

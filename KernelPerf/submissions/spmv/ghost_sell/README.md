# GHOST SELL-C-sigma submission

The `adapter.cu` implementation is intentionally small. GHOST itself is installed and pinned on
the CUDA worker under `/opt/ghost-cuda`; it is selected with the allow-listed
`ghost-cuda` build profile. The adapter converts the platform Host CSR into
GHOST SELL-C-sigma during `preprocess`, with `C=32` and the variant's sigma.

The example below evaluates sigma 128:

```bash
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32 \
  --submission submissions/spmv/ghost_sell \
  --configuration-id sigma-128
```

The directory contains one entry source for every sigma in
`{1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072}`;
`variants/index.json` is the machine-readable candidate configuration. The adapter
sorts rows by length inside each sigma-sized window during `preprocess`, while GHOST
builds and runs the C=32 SELL kernel. A CUDA scatter restores the original row order
after each solve, so solve-only timing includes the required correction. The worker
build must use the same
pinned GHOST and CUDA versions for all candidates.

The worker installation pins GHOST commit
`22a004dbbfb604d7b04b0bde813c8100d403cfcc`. The adapter widens the evaluator's
32-bit CSR indices to the worker's generated GHOST index ABI before creating
the sparse matrix, so the same submission also works with a GHOST build using
64-bit global or local indices.

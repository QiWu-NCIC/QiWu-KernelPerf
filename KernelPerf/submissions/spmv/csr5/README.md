# CSR5 multi-file submission

This directory demonstrates how to evaluate the public CUDA implementation
from [Benchmark_SpMV_using_CSR5](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5)
with KernelPerf. The files below `upstream/CSR5_cuda/` are kept as the upstream
source; only `adapter.cu` implements the KernelPerf lifecycle interface.

The adapter copies the platform-provided Device CSR before calling the public
`anonymouslibHandle` API because CSR5 preprocesses its column and value arrays
in place. The upstream API launches kernels on CUDA's legacy default stream,
so the adapter synchronizes that stream after each solve. This makes the
submission compatible with the current evaluator, while a future upstream
stream-aware API can remove that synchronization.

From `PlayGround/`, submit the directory through the same CLI used for every
other source submission (replace the API and backend with your deployment):

```powershell
python scripts/submit_spmv.py `
  --api http://127.0.0.1:18081 `
  --backend A100-SXM4-80GB `
  --dataset-id suitesparse_sample_100 `
  --operator spmv.csr.fp32 `
  --method-name CSR5-adapter `
  --base-format csr `
  --source-dir examples/csr5_submission `
  --entry-source adapter.cu `
  --wait
```

The resulting artifact has `kind: "source"`, `entry_source: "adapter.cu"`,
and no additional compile units because the CSR5 implementation is a
header-based template library. The web page offers the same fields: select **Source
directory**, choose this directory, set `base_format` to `CSR`, and select
`FP32` or `FP64` plus the desired `solve-only`, `pre+solve`, or
`pre/iteration+solve` timing mode.

The evaluator compiles `adapter.cu` as an independent plugin translation unit
and links it with the managed runner. The archived source package includes the
public header, CMake build, and a standalone caller under
`data/source/<job-id>/spmv/`.

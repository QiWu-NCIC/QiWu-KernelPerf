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

Evaluate the reviewed directory with the same CLI used for every source
submission:

```powershell
python -m kernelperf.cli evaluate `
  --config config/service.json `
  --backend A100-SXM4-80GB `
  --dataset-id suitesparse_sample_100 `
  --operator spmv.csr.fp32 `
  --submission submissions/spmv/csr5
```

The resulting artifact has `kind: "source"`, `entry_source: "adapter.cu"`,
and no additional compile units because the CSR5 implementation is a
header-based template library. Repeat the command with `spmv.csr.fp64` for
the FP64 build.

The evaluator compiles `adapter.cu` as an independent plugin translation unit
and links it with the managed runner. The databank source package includes the
public header, CMake build and standalone caller.

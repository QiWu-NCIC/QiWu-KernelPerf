# CSR5 multi-file submission

This directory demonstrates how to evaluate the corrected public CUDA
implementation from [CSR5 pull request #13](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/pull/13)
with KernelPerf. The files below `upstream/CSR5_cuda/` are fixed at commit
`caff9d80665052bd5379fb37fd135f0d72d7b525`, with the following patches:

- CUDA 12 synchronized shuffle intrinsics and guards for empty launches;
- bounds and synchronization fixes in partition-descriptor generation; and
- a stable segmented reduction that sums only the target segment instead of
  subtracting two warp-wide prefix sums; and
- an optional caller stream for SpMV launches, preserving default-stream
  behavior for existing callers.

The last change is required for signed, high-dynamic-range matrix values. The
upstream prefix-difference implementation can lose the entire target segment,
or produce `inf - inf`, when unrelated rows dominate the two prefixes.
`adapter.cu` implements the KernelPerf lifecycle interface without changing
the public CSR5 API.

The adapter copies the platform-provided Device CSR before calling the public
`anonymouslibHandle` API because CSR5 preprocesses its column and value arrays
in place. The adapter passes the evaluator stream to the upstream SpMV call,
so CUDA events measure the output reset and complete CSR5 solve without a host
synchronization in every iteration.

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

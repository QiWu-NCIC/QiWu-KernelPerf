# AlphaSparse CSR submission

`upstream/` is a complete snapshot of
[AlphaSparse/Library](https://github.com/AlphaSparse/Library) at commit
`39734b2d458a9a38059cd72795f3e64a8400e6f5`.
One non-algorithmic source patch moves a UTF-8 trailing comment in
`include/alphasparse/handle.h`; nvcc 12.x otherwise drops the following opening
brace on some hosts. Selected SpMV launches are also routed from the hard-coded
default stream to `handle->stream`. Kernel bodies, launch geometry, branches, reductions, and tuning
constants are unchanged.

The only implementation bridge is `adapter.cu`. It calls the official CUDA CSR
Scalar, Vector, Merge, LineEnhance, Flat1, Flat4, and Flat8
kernels directly. Temporary partition and merge buffers are allocated in
`qiwu_spmv_preprocess`. Every launch uses the CUDA stream supplied by the caller,
so KernelPerf CUDA events measure the GPU solve directly.

Build a standalone plugin with CUDA and CMake:

```bash
cmake -S . -B build -DQIWU_PLUGIN_ENTRY=variants/vector.cu
cmake --build build -j
./build/qiwu_spmv_example_fp32
./build/qiwu_spmv_example_fp64
```

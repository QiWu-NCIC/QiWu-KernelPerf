# AlphaSparse CSR submission

`upstream/` is a complete snapshot of [AlphaSparse/Library](https://github.com/AlphaSparse/Library)
with the CSR-Adaptive changes from [pull request #31](https://github.com/AlphaSparse/Library/pull/31)
at head commit `248af573c867cf4c7e05d40c611c7c34ebac4715`.
Integration changes are recorded in `provenance.json`. They include a source
encoding fix, routing selected launches through the caller stream, and the
CSR-Adaptive kernel-path changes evaluated by this submission. The adaptive
analysis and launch geometry remain upstream-compatible.

The only implementation bridge is `adapter.cu`. It calls the official CUDA CSR
Scalar, Vector, Adaptive, Merge, LineEnhance, Flat1, Flat4, and Flat8 kernels
directly. Adaptive row-block analysis and temporary workspace allocation run in
`qiwu_spmv_preprocess`. Every launch uses the stream supplied by the caller, so
KernelPerf events measure the GPU solve directly.

Build a standalone plugin with CUDA and CMake:

```bash
cmake -S . -B build -DQIWU_PLUGIN_ENTRY=variants/vector.cu
cmake --build build -j
./build/qiwu_spmv_example_fp32
./build/qiwu_spmv_example_fp64
```

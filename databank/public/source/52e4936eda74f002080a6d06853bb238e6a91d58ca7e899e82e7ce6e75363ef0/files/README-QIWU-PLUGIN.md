# QiWu rocSPARSE DTK 26.04 SpMV plugin

This directory is a standalone HIP source plugin and does not depend on the
KernelPerf Python service. It wraps the rocSPARSE v1 CSR SpMV API
(`rocsparse_spmv`). The worker supplies the HIP compiler/runtime and the
rocSPARSE library. It records the DTK 26.04 rocSPARSE 3.3.0 configuration
used for the Z100 regression. A standalone upstream rocSPARSE build requires
a compatible ROCm toolchain and dependencies and must be recorded separately.

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build build -j
./build/qiwu_rocsparse_example_fp32
./build/qiwu_rocsparse_example_fp64
```

`QIWU_PLUGIN_ENTRY` selects a packaged variant such as
`variants/adaptive.cu`. DTK 26.04 exposes the four configurations in this
sweep: `default`, `adaptive`, `stream`, and `lrb`. `adaptive` and `lrb` run
the rocSPARSE preprocess stage; `default` and `stream` only require
buffer-size and compute stages. The public interface is in
`include/qiwu/spmv_plugin.cuh`; applications call
`qiwu_spmv_preprocess`, `qiwu_spmv_solve`, and `qiwu_spmv_destroy`.

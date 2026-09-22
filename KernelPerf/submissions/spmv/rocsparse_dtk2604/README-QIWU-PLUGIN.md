# QiWu rocSPARSE DTK 26.04 SpMV plugin

This directory is a standalone HIP source plugin and does not depend on the
KernelPerf Python service. It wraps the DTK 26.04 rocSPARSE 3.3.0 generic
staged CSR SpMV API (`rocsparse_spmv`). A standalone upstream rocSPARSE build
requires a compatible ROCm toolchain and dependencies and must be recorded
separately.

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx906
cmake --build build -j
./build/qiwu_rocsparse_dtk2604_example_fp32
./build/qiwu_rocsparse_dtk2604_example_fp64
```

`QIWU_PLUGIN_ENTRY` selects a packaged variant such as
`variants/adaptive.cu`. This package exposes the three explicit CSR algorithms
in DTK 26.04: `adaptive`, `stream`, and `lrb`. The generic `default` selector is
excluded because it dispatches to an underlying explicit algorithm and is not
an independent kernel. `adaptive` and `lrb` run the rocSPARSE preprocess stage;
`stream` only requires the
buffer-size and compute stages. This matches the per-variant
`KERNELPERF_ROCSPARSE_NEEDS_PREPROCESS` definitions. The public interface is in
`include/qiwu/spmv_plugin.cuh`; applications call
`qiwu_spmv_preprocess`, `qiwu_spmv_solve`, and `qiwu_spmv_destroy`.

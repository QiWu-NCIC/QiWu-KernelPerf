# QiWu rocSPARSE SpMV plugin

This directory is a standalone HIP source plugin and does not depend on the
KernelPerf Python service. It wraps the rocSPARSE v1 CSR SpMV API
(`rocsparse_spmv`). The worker supplies the HIP compiler/runtime and the
rocSPARSE library. For the BW1000 regression this is the pinned DTK 26.04
installation; a standalone upstream rocSPARSE build requires a compatible
ROCm toolchain and dependencies and must be recorded separately.

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build build -j
./build/qiwu_rocsparse_example_fp32
./build/qiwu_rocsparse_example_fp64
```

`QIWU_PLUGIN_ENTRY` selects a packaged variant such as
`variants/adaptive.cu`. The sweep keeps the common candidate names `default`,
`adaptive`, `stream`, `lrb`, and `nnzsplit`. In DTK 26.04, the v1 header
exposes `default`, `csr_adaptive`, `csr_stream`, and `csr_lrb`; `nnzsplit` is only available in newer
rocSPARSE headers and is reported as unavailable when the installed header
does not define it. `adaptive`, `lrb`, and `nnzsplit` run the v1 preprocess
stage; `default` and `stream` only require buffer-size and compute stages. The
public interface is in
`include/qiwu/spmv_plugin.cuh`; applications call
`qiwu_spmv_preprocess`, `qiwu_spmv_solve`, and `qiwu_spmv_destroy`.

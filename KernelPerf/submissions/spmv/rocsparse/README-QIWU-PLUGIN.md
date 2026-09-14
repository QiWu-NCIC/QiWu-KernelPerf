# QiWu rocSPARSE SpMV plugin

This directory is a standalone HIP source plugin and does not depend on the
KernelPerf Python service. It wraps the public rocSPARSE CSR descriptor API.
The benchmark worker builds and links an explicitly checked-out upstream
rocSPARSE source tree; it does not use the worker's packaged rocSPARSE
library. The HIP runtime/compiler required by the target accelerator remains a
worker deployment concern.

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build build -j
./build/qiwu_rocsparse_example_fp32
./build/qiwu_rocsparse_example_fp64
```

`QIWU_PLUGIN_ENTRY` selects a packaged variant such as
`variants/adaptive.cu`. The available CSR candidates are `default`,
`adaptive`, `rowsplit`, `lrb`, and `nnzsplit`. The old `stream` spelling is
not included because upstream rocSPARSE treats it as a deprecated alias of
`rowsplit`. The public interface is in
`include/qiwu/spmv_plugin.cuh`; applications call
`qiwu_spmv_preprocess`, `qiwu_spmv_solve`, and `qiwu_spmv_destroy`.

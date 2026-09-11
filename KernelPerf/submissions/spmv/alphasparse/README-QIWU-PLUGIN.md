# Qiwu SpMV source plugin

This directory is a standalone CUDA source plugin. It does not require KernelPerf.

```bash
cmake -S . -B build
cmake --build build -j
./build/qiwu_spmv_example_fp32
./build/qiwu_spmv_example_fp64
```

`plugin.json` is the package manifest. It records the payload files, content hash,
evaluated entry source, configurations, and dependencies. CMake uses its
`entry_source` by default. Select another packaged variant with
`-DQIWU_PLUGIN_ENTRY=variants/name.cu`. The `upstream/` directory is the complete
AlphaSparse/Library snapshot. Its CUDA kernels are unchanged; the only source
syntax patch moves a UTF-8 comment in `handle.h` so nvcc preserves the following
brace. The selected SpMV launch sites are routed through `handle->stream` without
changing launch geometry or kernel logic. `adapter.cu` calls those official CUDA
SpMV kernels directly.

Applications include `include/qiwu/spmv_plugin.cuh`, link either
`qiwu_spmv_fp32` or `qiwu_spmv_fp64`, and call `qiwu_spmv_preprocess`,
`qiwu_spmv_solve`, and `qiwu_spmv_destroy`. `examples/standalone.cu` is the
smallest complete caller.

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
`-DQIWU_PLUGIN_ENTRY=variants/name.cu`. External libraries
can be supplied with `QIWU_PLUGIN_EXTRA_INCLUDE_DIRS` and
`QIWU_PLUGIN_EXTRA_LIBRARIES`. GHOST packages accept `-DGHOST_ROOT=/path/to/install`.

Applications include `include/qiwu/spmv_plugin.cuh`, link either
`qiwu_spmv_fp32` or `qiwu_spmv_fp64`, and call `qiwu_spmv_preprocess`,
`qiwu_spmv_solve`, and `qiwu_spmv_destroy`. `examples/standalone.cu` is the
smallest complete caller.

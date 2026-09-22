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
AlphaSparse/Library snapshot with the CSR-Adaptive changes from
[pull request #31](https://github.com/AlphaSparse/Library/pull/31) at head commit
`248af573c867cf4c7e05d40c611c7c34ebac4715`. Integration changes are listed
in `provenance.json`. The CSR-Adaptive HIP and CUDA headers contain the small
kernel-path changes tested on BW1000: row offsets are cached per
workgroup, the full-block CSR-Stream load uses a bounds-safe fast path,
invalid row-pointer reads are kept behind the row guard, and beta-zero output
stores avoid reading the old output. The adapter invokes the upstream kernel
directly and keeps preprocessing outside solve-only timing.

Applications include `include/qiwu/spmv_plugin.cuh`, link either
`qiwu_spmv_fp32` or `qiwu_spmv_fp64`, and call `qiwu_spmv_preprocess`,
`qiwu_spmv_solve`, and `qiwu_spmv_destroy`. `examples/standalone.cu` is the
smallest complete caller.

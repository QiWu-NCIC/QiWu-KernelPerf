# Standalone plugins

Each reviewed method is a directory under
`KernelPerf/submissions/<operator>/<method>/`. The directory is both the
evaluator input and the source package published by the databank.

## Required files

```text
<method>/
|-- submission.json
|-- CMakeLists.txt
|-- include/qiwu/<operator>_plugin.cuh
|-- adapter.cu (or variants/<configuration>.cu)
|-- examples/standalone.cu
`-- README-QIWU-PLUGIN.md
```

Upstream source and license files may be included when the adapter depends on
them. The package manifest generated for downloads records every file and a
content hash. Do not put worker paths, credentials, datasets or evaluator
Python modules in a plugin.

## SpMV contract

The public header receives CSR data owned by the caller. `preprocess` may copy
or transform that data into any device-side format; that work is outside the
solve timer. `solve` performs only `y = A * x`; repeated calls must reuse the
prepared state. `destroy` releases all resources. The same source is compiled
twice for FP32 and FP64 (FP64 builds define `QIWU_SPMV_FP64=1`).

```cpp
qiwu_spmv_preprocess(...);
qiwu_spmv_solve(...);
qiwu_spmv_destroy(...);
```

The exact types and error contract are defined by the public header shipped in
the plugin. Do not reintroduce the removed `kernelperf_spmv_*` or `abi_version`
interfaces.

## Local build

```bash
cmake -S submissions/spmv/<method> -B /tmp/qiwu-plugin-build
cmake --build /tmp/qiwu-plugin-build -j
/tmp/qiwu-plugin-build/qiwu_spmv_example_fp32
/tmp/qiwu-plugin-build/qiwu_spmv_example_fp64
```

For a multi-configuration package, select an entry source using the CMake
option documented by that package. A relocatable `.o` is accepted only when it
was compiled for the target worker architecture and exports the same public
contract. Source submissions are preferred because they remain portable across
the supported CUDA versions.

## Adapter guidance

Keep upstream calls in the adapter and isolate compatibility shims in a small
file. Put allocation, format conversion, sorting, and library descriptors in
`preprocess`; do not hide them in the timed solve path. Record the upstream
repository, commit or release, CUDA requirement and configuration ID in
`submission.json` and the method README. Every sweep configuration must have a
unique `configuration_id` and a shared `candidate_group` when it participates
in BEST selection.

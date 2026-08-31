# SpMV plugin reference

The maintained SpMV plugin contract is documented in
[`docs/PLUGINS.md`](../docs/PLUGINS.md). A plugin directory contains its public
header, adapter, optional upstream sources, `submission.json`, CMake example
and standalone build instructions. The evaluator and downloaded package use
the same files.

The source-level lifecycle is:

```cpp
qiwu_spmv_preprocess(...);
qiwu_spmv_solve(...);
qiwu_spmv_destroy(...);
```

Inputs are host/device CSR data. Preprocessing may build a custom device
format; solve is timed as SpMV only. FP32 and FP64 are separate builds.

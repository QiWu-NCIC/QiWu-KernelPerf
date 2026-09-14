# Contributing

## Submission directory

Add one reviewed method under `KernelPerf/submissions/<operator>/<method>/`.
For SpMV, use `KernelPerf/submissions/spmv/<method>/`. This directory is the
evaluator input and should contain only the manifest, the adapter, and the
source files required to build the submission. Upstream code and licenses may
be included; do not include worker paths, credentials, datasets, or evaluator
Python modules.

```text
<method>/
|-- submission.json
|-- adapter.cu (or the entry_source named in submission.json)
|-- variants/<configuration>.cu     # optional, for configuration sweeps
|-- include/                        # optional, method-owned headers
`-- upstream/                       # optional, vendored implementation
```

`submission.json` is required. Its `entry_source` defaults to `adapter.cu`.
Additional translation units and include directories must be declared with
`compile_units` and `include_dirs`; every referenced file must be inside the
submission directory. The reserved paths
`include/qiwu/<operator>_plugin.cuh`, `examples/standalone.cu`,
`CMakeLists.txt`, and `README-QIWU-PLUGIN.md` are generated for the downloadable
standalone package and should not be submitted as evaluator input.

SpMV plugins expose `qiwu_spmv_preprocess`, `qiwu_spmv_solve`, and
`qiwu_spmv_destroy`. Preprocessing may build device storage and is outside the
solve timer; solve only computes `y = A * x`. The same source is compiled for
FP32 and FP64 (`QIWU_SPMV_FP64=1`). CUDA and HIP submissions use
`language: "cuda"` or `language: "hip"`; a portable adapter may declare both in
`languages`. The contract's runtime shim maps evaluator-owned operations to the
selected backend. Every sweep configuration needs a unique
`configuration_id` and stable `method_id`.

The databank packages an accepted source tree as an independent plugin. Build
that generated/downloaded package without importing KernelPerf:

```bash
cmake -S <downloaded-plugin> -B /tmp/qiwu-plugin-build
cmake --build /tmp/qiwu-plugin-build -j
/tmp/qiwu-plugin-build/qiwu_spmv_example_fp32
/tmp/qiwu-plugin-build/qiwu_spmv_example_fp64
```

Relocatable `.o` submissions are accepted only for a declared, matching GPU
architecture. Source submissions are preferred for portability.

## Pull requests

Open a PR containing the submission directory. GitHub Actions performs schema,
path, size, and compile checks; it does not connect to evaluation servers. After
review, a maintainer checks out the immutable PR commit on an isolated worker,
runs the evaluator, reviews the CSVs, and updates the databank separately.
Execution uses a restricted account, read-only checkout, no network access and
explicit process and time limits.

Before opening the PR, run:

```bash
cd KernelPerf
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_submissions.py submissions
```

Keep upstream calls in the adapter and compatibility shims small. Record the
upstream repository and revision, CUDA or HIP requirements, configuration IDs and
license in `submission.json` and the method README.

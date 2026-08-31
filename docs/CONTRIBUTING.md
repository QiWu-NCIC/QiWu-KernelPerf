# Contributing

## Submission package

Add one reviewed method under `KernelPerf/submissions/<operator>/<method>/`.
The directory is both evaluator input and the standalone source package shown
in the databank. It must contain `submission.json`, `CMakeLists.txt`, the public
plugin header, an adapter, a standalone example, and `README-QIWU-PLUGIN.md`.
Upstream code and licenses may be included; do not include worker paths,
credentials, datasets, or evaluator Python modules.

```text
<method>/
|-- submission.json
|-- CMakeLists.txt
|-- include/qiwu/<operator>_plugin.cuh
|-- adapter.cu (or variants/<configuration>.cu)
|-- examples/standalone.cu
`-- README-QIWU-PLUGIN.md
```

SpMV plugins expose `qiwu_spmv_preprocess`, `qiwu_spmv_solve`, and
`qiwu_spmv_destroy`. Preprocessing may build device storage and is outside the
solve timer; solve only computes `y = A * x`. The same source is compiled for
FP32 and FP64 (`QIWU_SPMV_FP64=1`). Every sweep configuration needs a unique
`configuration_id` and stable `method_id`.

Build a package independently of KernelPerf:

```bash
cmake -S KernelPerf/submissions/spmv/<method> -B /tmp/qiwu-plugin-build
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
upstream repository and revision, CUDA requirements, configuration IDs and
license in `submission.json` and the method README.

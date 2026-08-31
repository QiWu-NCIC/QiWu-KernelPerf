# KernelPerf SpMV CSV Schema

`schema_version=2` identifies the `kernelperf-spmv-v2` row contract. Version 2 names the efficiency field `solve_only_efficiency_percent` so its timing scope is explicit. The schema version is independent of CUDA, cuSPARSE, and submission versions.

`operations` is the operation count used for the reported performance. For SpMV it is `2 * nnz`. `solve_gflops = operations / (solve_ms * 1e6)`. `solve_only_efficiency_percent = solve_gflops / peak_gflops * 100`; the field name explicitly records that this efficiency uses `solve_ms` only. `preprocess_ms` is reported separately and is included only when the UI time mode explicitly selects `pre+solve` or `pre/iteration+solve`.

The result filename is `method_id-backend_id-dataset_id-dtype.csv`; `submission_id` remains in each row for provenance but is not part of the retained filename. Public results and candidate sweeps are additionally partitioned by `operator/backend/dataset` directories, so files from different datasets cannot share a directory or a curation key.

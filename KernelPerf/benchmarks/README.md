# Adding a benchmark

Each benchmark is an independent Python package under `benchmarks/<name>/` and owns its execution artifacts:

- `driver.py`: a `kernelperf.benchmark.Benchmark` implementation that validates submissions and converts one worker execution into `BenchmarkResult` records.
- `template.*`: an optional administrator-owned candidate template or harness when the driver needs one.

To add one:

1. Create `benchmarks/<name>/__init__.py` and `driver.py`; add `template.*` when the benchmark uses a managed template or harness.
2. Subclass `Benchmark`; keep compilation, execution, result parsing, and correctness validation inside that directory.
3. Add one entry to `config/benchmarks.json` with `benchmark_id`, `driver` (`module:class`), operators, defaults, and driver options; set `template` only when present.
4. Reference datasets only by IDs declared in `config/datasets.json`; never embed dataset names or worker IDs in driver code.
5. Add unit tests for submission validation, result classification, and error metadata. The core scheduler discovers the benchmark from configuration and requires no code change.

`spmv/` is the complete reference: external users submit only the configured kernel function, while the driver and CUDA template own data loading, timing, and CPU-reference validation.

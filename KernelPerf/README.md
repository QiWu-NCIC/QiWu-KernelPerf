# KernelPerf evaluator

This directory contains the maintainer-run Python evaluator. Start with the
repository [README](../README.md), then read [`docs/CONTRIBUTING.md`](../docs/CONTRIBUTING.md)
for the source contract and [`docs/OPERATIONS.md`](../docs/OPERATIONS.md) for
worker execution.

Install and test from this directory:

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_submissions.py submissions

# Evaluate reviewed submissions for one backend and dataset
python scripts/run_regression.py \
  --config config/service.json \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100
```

The CLI evaluates reviewed submissions and writes deterministic CSV exports;
there is no HTTP submission server in this repository.

`config/service.json` is a portable local example. Site-specific worker and
dataset profiles belong in the ignored `config/private/` directory.

## Measurement protocol

SpMV uses 5 untimed warmups and 20 timed solves per matrix and dtype. Standard
solve timing uses one CUDA Event interval around the complete repeat loop;
preprocessing is measured separately with a host clock through stream
synchronization. Correctness is checked by a separate solve against the CPU
CSR reference. For row `i`, with `n_i` nonzeros, `s_i =
sum_j(abs(a_ij*x_j))` and `u = eps(storage_type)/2`, the dynamic pass bound is
`abs(y-y_ref)/s_i <= 4*n_i*u`; the tested storage types are `float` (FP32) and
`double` (FP64). Both dtypes use a host `long double` accumulator; products are
promoted to `long double` before accumulation. Rows
whose condition estimate makes `n_i*u*kappa_i >= 1` are reported as numerical
noise rows and excluded from aggregate error statistics. The authoritative configuration is
[`config/benchmarks.json`](config/benchmarks.json); lifecycle timing and the
reference implementation are in
[`benchmarks/spmv/template.cu`](benchmarks/spmv/template.cu), and the runtime
configuration handoff is in [`benchmarks/spmv/driver.py`](benchmarks/spmv/driver.py).

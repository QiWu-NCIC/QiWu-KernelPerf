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

## Measurement protocols

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

SpMM evaluates CSR `A * B` with row-major dense matrices, `op(A)=N`, `op(B)=N`,
`alpha=1`, and `beta=0`. Main rankings use RHS widths `{2,4,8,16,32,64,128}`;
`N=1` is sanity-only. The driver records solve-only, preprocess-plus-solve,
and amortized preprocess-plus-solve timings, with `2 * nnz * N` real FLOPs.
FP32 and FP64 use `float` and `double` outputs respectively and validate
against a host `long double` reference with the dynamic row-length bound and
safety factor `C=4`. The independent lifecycle is defined in
[`benchmarks/spmm/template.cu`](benchmarks/spmm/template.cu), with CSR FP32 and
FP64 registered in [`config/benchmarks.json`](config/benchmarks.json).

## Resumable full campaigns

`scripts/run_spmv_campaign.py` and `scripts/run_spmm_campaign.py` checkpoint
each matrix in SQLite and export candidate CSVs after each configuration.
Run them from this directory on an allocated GPU worker, inside tmux or a
scheduler job. Use `--shard-index` and `--shard-count` to partition the manifest
across workers; each shard needs its own database and build directory.

`scripts/run_full_campaign.py` creates a shard-specific service profile from
a template containing `__SHARD__` in state paths. It runs SpMV before SpMM on
the matrices currently available and returns code 75 while a dataset shard is
still downloading. A persistent supervisor can rerun the same command to
resume recorded work. Completion markers apply to individual shards, rather
than the entire platform.

The default resume mode preserves recorded failures. Use `--retry-failures`
for an explicit retry while retaining prior attempts in the database. Source,
protocol and dataset fingerprints must match when resuming. A changed
toolchain or implementation requires a new campaign ID; `--adopt-fingerprint`
is reserved for reviewed migrations and does not revalidate existing results.
Keep site profiles, downloaded matrices, databases and operational snapshots
outside the published repository.

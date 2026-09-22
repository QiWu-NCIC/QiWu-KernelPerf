# SpMM P0 Development and Evaluation Report

Date: 2026-09-23

## Scope

This report tracks the first QiWu-KernelPerf CSR SpMM campaign. The P0 scope
contains an independent SpMM plugin contract, evaluator, correctness checker,
CSV schema, source packaging, result databank, static web view, and adapters
for AlphaSparse, cuSPARSE, and rocSPARSE DTK 26.04.

## Implementation

- Contract: `KernelPerf/include/qiwu/spmm_plugin.cuh`.
- Driver: `KernelPerf/benchmarks/spmm/`.
- Reviewed sources: `KernelPerf/submissions/spmm/`.
- Result schema: `kernelperf-spmm-v3`.
- Databank catalog: `databank/public/data/spmm/index.json`.
- Web view: `databank/src/views/SPMM/index.vue`.

The SpMM lifecycle is separate from SpMV. Adapters own only upstream
descriptor creation, workspace management, preprocess dispatch, solve dispatch,
version reporting, and cleanup. CSR5, CSR-Adaptive, and GHOST SELL-C-sigma are
not represented because their reviewed sources do not expose a general SpMM
operation.

## Measurement Protocol

- Dataset: `suitesparse_sample_100`.
- Dtypes: FP32 and FP64.
- Sparse format: CSR.
- RHS columns: `N={2,4,8,16,32,64,128}`.
- Sanity-only RHS columns: `N=1`.
- Dense layout: row-major `B` and `C`.
- Operations: `op(A)=N`, `op(B)=N`.
- Scalars: `alpha=1`, `beta=0`.
- Input seed: `20260922`.
- Sampling: 5 warmups and 20 measured iterations.
- FLOP count: `2 * nnz * N`.

Every result stores solve-only, preprocess-plus-solve, and amortized
preprocess-plus-solve timing, corresponding GFLOP/s and efficiency, validation
status, failure counts, explicit error type/failure stage, runtime versions,
and the selected configuration for derived BEST rows. BEST selection is scoped
by platform, dataset, dtype, RHS columns, dense layout, candidate group, and
timing metric.

## Correctness

FP32 output uses `float`; FP64 output uses `double`. The host reference promotes
each product to `long double` before accumulation. Each `C[i,q]` is checked
with the dynamic row-length error rule and safety factor `C=4`. Dimension
errors, runtime errors, precision errors, NaN values, and Inf values are
reported separately. `KernelPerf/tests/fixtures/spmm_incorrect.cu` intentionally
produces invalid output and is required to fail validation.

## Locked Sources and Libraries

- AlphaSparse adapter source commit:
  `248af573c867cf4c7e05d40c611c7c34ebac4715`.
- AlphaSparse upstream base commit:
  `39734b2d458a9a38059cd72795f3e64a8400e6f5`.
- Local CUDA smoke: CUDA 12.5, cuSPARSE 12.5.1.
- H100 probe: CUDA 12.8.2, cuSPARSE 12.5.8.
- RTX5090 probe: CUDA 12.8, driver 595.84; AlphaSparse CUDA `sm_120`
  compilation failures are retained as unavailable candidates.
- A100 probe: CUDA 12.2, driver 580.178.04.
- BW1000 probe: DTK 26.04, HIP runtime `60326113`,
  rocSPARSE 3.3.0 revision `08e9b279`.

Library versions are also embedded in public method names and CSV metadata.

## Validation Status

- Local Python tests: 82 passed.
- Reviewed submission validation: passed for all SpMV and SpMM manifests.
- Databank production build: passed with Node.js 24.19.0.
- Static databank server: `http://127.0.0.1:8080/`, including SpMV/SpMM tabs,
  scoped filters, per-CSV download, all-result archive, and source download.
- Real local GPU chain: cuSPARSE CSR ALG1 passed and completed CSV import/audit.
- H100 smoke: cuSPARSE CSR ALG1 passed.
- BW1000 smoke: rocSPARSE CSR staged API passed.
- BW1000 AlphaSparse smoke: ALG1, ALG3, ALG4, and ALG5 passed; ALG2 produced
  precision-error rows for some RHS counts and is retained as a failed
  candidate rather than rewritten or dropped.

## Platform Campaign

| Platform | Applicable adapters | Status |
| --- | --- | --- |
| A100-SXM4-80GB | AlphaSparse CUDA, cuSPARSE | Authenticated; full checkpoint running |
| H100-SXM5-80GB | AlphaSparse CUDA, cuSPARSE | Full campaign running in Slurm |
| RTX5090-SL3061 | AlphaSparse CUDA, cuSPARSE | Authenticated; full checkpoint running; AlphaSparse `sm_120` compile failures retained |
| BW1000-gfx936 | AlphaSparse HIP, rocSPARSE | Full checkpoint running; AlphaSparse ALG2 precision failures retained |
| Z100-gfx906 | AlphaSparse HIP, rocSPARSE | Fresh dynamic token pending |

Remote control shells use `tmux`; H100 and BW1000 computation runs under
Slurm. Run directories, logs, SQLite databases, raw CSVs, and exports are kept
under fresh per-campaign directories so interrupted sessions can be inspected
without overwriting historical results.
`KernelPerf/scripts/run_spmm_campaign.py` additionally checkpoints per matrix,
uses a deterministic campaign identity, verifies a source/protocol/dataset/
worker fingerprint, and resumes only missing `(candidate_group,configuration,
matrix,N)` rows. This keeps identical configuration names in different adapter
families independent.

## Known Issues and P1 Work

- Complete the five-platform sample100 campaign and import every raw CSV.
- Audit and manually sample BEST selections for all three timing metrics.
- Investigate AlphaSparse HIP ALG2 precision failures with upstream maintainers.
- Add P1 formats only when an upstream general SpMM API is available.
- Evaluate column-major dense layouts and transpose operations as separate
  ranking scopes rather than mixing them with P0.

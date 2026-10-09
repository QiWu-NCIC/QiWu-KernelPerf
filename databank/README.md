# QiWu Databank

Static leaderboard for the unified QiWu-KernelPerf repository. Reviewed source
lives only in `../KernelPerf/submissions`; `public/source` is a generated build
directory and is not a second source tree or a versioned data directory.

```bash
npm ci
npm run audit:all        # also generates public/source
npm run dev
```

Node.js 20+ is required. The browser uses geometric means for aggregate GFLOP/s,
efficiency, preprocess time, solve time and effective time, and applies the 90%
coverage rule. A method below 90% successful coverage remains visible with
failure counts but is marked `Unranked`. Known datasets use the declared
`matrix_count` in `KernelPerf/config/datasets.json` as the coverage denominator,
including matrices and dtypes whose results have not arrived yet. Partial
full-corpus checkpoints remain visible and downloadable without implying that
the campaign is complete. Platform CPU/GPU labels are maintained
in `public/data/platforms.json`.

The protocol shown on the leaderboard is authoritative: 5 untimed warmups, 20
timed solves, and a separate CPU CSR correctness solve. The dynamic row-error
bound is `abs(y-y_ref)/s_i <= 4*n_i*u`, where `n_i` is the row length and `u`
comes from the actual `float` or `double` storage. Both FP32 and FP64 use a host
`long double` reference; products are promoted to `long double` before accumulation.
Numerical-noise rows are
reported separately. Its formulas, timing boundaries and implementation locations are documented in the repository
[`README.md`](../README.md#spmv-measurement-protocol).

The SpMM view combines FP32 and FP64 efficiency like SpMV, and filters by
platform, dataset, RHS columns, base format, and timing scope. Its catalog is stored at
`public/data/results/spmm/index.json` alongside candidate and BEST CSVs, which
remain partitioned by RHS count and dense layout:

```text
public/data/results/spmm/<backend>/<dataset>/<dtype>/n<N>/<layout>/<method>/<submission>.csv
```

Use `npm run add:spmm -- result.csv --source-dir
public/source/spmm-baselines/<plugin>` to import reviewed data and
`npm run audit:spmm` to verify identities, source hashes, metrics, coverage,
and BEST selections.

Result files are partitioned by `operator/backend/dataset`:

```text
public/data/results/<operator>/<backend>/<dataset>/<method>-<backend>-<dataset>-<dtype>.csv
public/data/candidate-pool/<operator>/<backend>/<dataset>/<method>-<backend>-<dataset>-<dtype>.csv
```

The JSON manifests are authoritative and are committed with the CSV files so
GitHub Pages can download them directly. `npm run dev` and `npm run build`
automatically generate `public/source` from the canonical submissions. See
[`docs/DATA.md`](../docs/DATA.md) for the data policy.

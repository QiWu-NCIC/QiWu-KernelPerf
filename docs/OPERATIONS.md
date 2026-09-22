# Maintainer Operations

`KernelPerf/kernelperf` is the evaluator and scheduler. `KernelPerf/submissions`
contains reviewed plugins. `KernelPerf/data` is ignored runtime state.
`databank/public/data` is the versioned CSV catalog. `databank/public/source`
is an ignored build directory generated from canonical submissions.

## Evaluate and publish

From `KernelPerf/`, evaluate every reviewed submission for one platform and
dataset with the parameterized regression runner:

```bash
python scripts/run_regression.py \
  --config config/private/service-a100.json \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100
```

Use `--submission` and repeated `--operator` options to narrow a run. The
portable `config/service.json` is only a local example; platform profiles are
kept under the ignored `config/private/` directory. Isolate
long sweeps with independent SQLite and export directories; preserve rows for
failed matrices. Exports are written below
`KernelPerf/data/result_exports/<suite>/<backend>/<dataset>/`.

For the fixed CSR SpMM campaign, run the checkpointed driver inside `tmux` or
one Slurm allocation:

```bash
python scripts/run_spmm_campaign.py \
  --config config/private/service-h100.json \
  --backend H100-SXM5-80GB \
  --dataset-id suitesparse_sample_100 \
  --campaign-id h100-spmm-p0-20260922-v1 \
  --phase all
```

The campaign ID deterministically identifies its jobs. The runner verifies a
source/protocol/dataset/worker fingerprint, checkpoints after each matrix,
exports partial CSVs, and skips existing `(candidate-group,configuration,matrix,N)`
rows on restart.
Exit code 75 means the time budget or a termination signal produced a safe
checkpoint and the same command can resume.

Copy accepted results into the matching databank scope. Use
`npm run add:spmv -- <file.csv>` for a single public result, or
`--candidate` for a configuration sweep. For a CSV pull request, first run
`npm run check:result-submissions`; import the accepted inbox file with a
generated standalone package, then remove the inbox copy:

```bash
cd databank
npm run prepare:sources
npm run check:result-submissions
npm run add:spmv -- result-submissions/spmv/<file.csv> \
  --source-dir public/source/baselines/<submission>
rm result-submissions/spmv/<file.csv>
```

Use `npm run add:spmm -- <file.csv> --source-dir
public/source/spmm-baselines/<submission>` for reviewed SpMM CSVs. Keep each
RHS count and dense layout in its own CSV and run `npm run audit:spmm` after
import.

Then run:

```bash
cd databank
npm run audit:spmv
npm run build
```

Commit CSV files and JSON indexes together. Before local development, audits or
deployment, the databank scripts generate standalone source packages from
`KernelPerf/submissions`; the browser then performs GFLOP/s, efficiency,
geometric-mean, coverage and BEST calculation from the GitHub-hosted CSV files.

## Platform notes

Use the private worker profiles under `KernelPerf/config/private/` and record
the exact worker label, CUDA toolkit or HIP runtime version, dataset manifest
hash, commit SHA, configuration IDs and pass/fail
counts. Credentials, temporary fallback profiles and downloaded matrix archives
stay outside Git. Use a Slurm GPU allocation on platforms whose login node has
no GPU, and give parallel batches independent SQLite and export paths.

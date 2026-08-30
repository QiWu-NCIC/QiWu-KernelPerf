# Deployment and operations

## Worker-pinned SpMV libraries

Run `scripts/install_spmv_worker_dependencies.sh` as root on the worker before
enabling the `ghost-cuda` build profile. It pins GHOST, builds it with 32-bit
global/local indices, and writes
`~/.config/kernelperf/spmv-dependencies.json`. The script uses the worker's
pinned CUDA 12.8 toolkit and
fails rather than mixing toolkit versions.

All non-secret deployment values are defined in `config/deployment.json`; service and worker runtime settings are referenced from the other files in `config/`.

## Entry service

Set the existing administrator secret. To enable the static leaderboard publish
button, also set a GitHub token with content write access to the configured
databank branch:

```bash
export KERNELPERF_ADMIN_TOKEN='<secret>'
export KERNELPERF_GITHUB_TOKEN='<github-token>'
python3 scripts/deploy_kernelperf_entry.py
```

The deployer:

1. queries the configured entry from inside its SSH session and waits until queued/running work is empty;
2. creates online backups of both SQLite databases under `backups/`;
3. uploads a source archive without caches, databases, generated data, or third-party repositories;
4. replaces the managed `benchmarks`, `config`, `kernelperf`, `scripts`, and `web` trees;
5. installs the package, restarts the service, and waits for `/api/v1/workers` to return HTTP 200.

It deliberately preserves runtime databases, backups, datasets, virtual
environments, third-party benchmark repositories, automatic result CSVs under
the configured `result_exports` directory, submitted source snapshots under
the configured `source_submissions` directory, and both service tokens from the
previous process when they are not supplied again.

## Automatic result CSVs

After a job reaches a terminal state, KernelPerf writes one
canonical CSV for each SpMV backend/operator/method selection. The default
directory is `data/result_exports`, arranged as:

```text
data/result_exports/spmv/<backend-id>/<submission-id>.csv
```

SQLite remains the authoritative runtime store. CSV files are written through
a temporary file and atomically replaced. An export failure is recorded in the
job log without changing the benchmark status. A terminal job with no stored
result rows does not create an empty CSV.

## Web proxy

```bash
python3 scripts/deploy_web_proxy.py
```

The script copies the proxy runtime and recreates only the configured proxy container. Container image, network, ports, application container, and target address all come from deployment configuration.

## Verification

After deployment, verify:

```bash
curl "$(python3 -c 'import json; print(json.load(open("config/deployment.json"))["entry"]["public_url"])')/api/v1/workers"
curl "$(python3 -c 'import json; print(json.load(open("config/deployment.json"))["entry"]["public_url"])')/api/v1/queue"
```

Expected invariants are: every configured worker is present, the queue has `queued_tasks` plus aggregate `queued_job_ids`, idle workers have no `current_job_id`, and both SQLite databases pass `pragma integrity_check`.

## Contest and task operations

```bash
python3 scripts/manage_contests.py list
python3 scripts/manage_contests.py add \
  --contest-id ID --name NAME --start-time ISO8601 \
  --backend-id BACKEND --dataset-id DATASET --benchmark BENCHMARK \
  --operator-id OPERATOR
python3 scripts/submit_task.py --payload request.json --admin --wait
```

No operational script selects a worker, dataset, endpoint, or benchmark by a built-in default. Those choices are explicit payload fields or configuration values.

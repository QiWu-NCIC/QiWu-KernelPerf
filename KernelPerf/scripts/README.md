# Operational scripts

- `deploy_kernelperf_entry.py`: waits for an empty queue, backs up both SQLite databases, installs the configured release, and restarts the entry service.
- `deploy_web_proxy.py`: deploys the configured TCP web proxy container.
- `manage_contests.py`: lists, creates, and deletes administrator-owned contests through the REST API.
- `submit_task.py`: submits an arbitrary job payload and can wait for completion.
- `kernelperf_web_proxy.py`: the proxy runtime copied by `deploy_web_proxy.py`.

Addresses, SSH targets, ports, paths, worker definitions, datasets, benchmarks, and runtime parameters are read from `config/`. Secrets remain in environment variables.

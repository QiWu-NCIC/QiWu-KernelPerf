# Private Deployment Profiles

This directory is for maintainer-only deployment files and is excluded from
Git. Provision these files from the operations secret store or a protected
machine backup:

```text
service-a100.json
service-h100.json
service-rtx5090.json
workers-a100.json
workers-h100.json
workers-rtx5090.json
datasets-a100.json
datasets-h100.json
datasets-rtx5090.json
```

The service files should reference `config/benchmarks.json` and the private
worker/dataset files with paths relative to `KernelPerf`. Worker files contain
hostnames, SSH settings, CUDA/GHOST installation paths and scheduler options;
dataset files contain the mounted matrix root. Never commit those values or
private keys.

# Configuration

The tracked JSON files are portable examples. They contain no site addresses,
credentials, CUDA installation paths, dataset mount points or absolute build
paths. The default
`service.json` describes one local CUDA worker and expects matrices below
`KernelPerf/data/datasets/suitesparse`.

Maintainer platform profile JSON files belong in the ignored `config/private/`
directory.
Copy the checked-out private profiles there, then select one explicitly:

```bash
python scripts/run_regression.py \
  --config config/private/service-h100.json \
  --backend H100-SXM5-80GB \
  --dataset-id suitesparse_sample_100
```

Private profiles should reference paths relative to `KernelPerf` where
possible. Keep SSH keys, hostnames, scheduler settings, CUDA/GHOST prefixes and
dataset roots out of Git.

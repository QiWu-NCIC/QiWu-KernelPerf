#!/usr/bin/env bash
. /etc/profile >/dev/null 2>&1 || :
set -euo pipefail
cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
module load cuda/12.8.2
export PYTHONPATH="$HOME/yjk/pydeps:$HOME/yjk/QiWu-KernelPerf/KernelPerf"
rm -rf data/result_exports/h100 data/source/h100 data/kernelperf-h100.sqlite data/contest-h100.sqlite
mkdir -p data
PYTHONPATH=. python3 -m kernelperf.cli evaluate --config config/service-h100.json \
  --backend H100-SXM5-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32 \
  --submission submissions/spmv/cusparse \
  --timeout 1200

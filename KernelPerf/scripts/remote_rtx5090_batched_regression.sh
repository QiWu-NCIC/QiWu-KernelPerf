#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH="$HOME/yjk/pydeps:$HOME/yjk/QiWu-KernelPerf/KernelPerf"
mkdir -p data
rm -rf data/result_exports/rtx5090 data/source/rtx5090
rm -f data/kernelperf-rtx5090.sqlite

run() {
  local submission=$1 operator=$2 config_id=${3:-}
  echo "===== $submission $operator ${config_id:-all} =====" >&2
  if [[ -n "$config_id" ]]; then
    python3 -u -m kernelperf.cli evaluate \
      --config config/service-rtx5090.json \
      --submission "$submission" --backend RTX5090-SL3061 \
      --dataset-id suitesparse_sample_100 --operator "$operator" \
      --configuration-id "$config_id" --timeout 14000
  else
    python3 -u -m kernelperf.cli evaluate \
      --config config/service-rtx5090.json \
      --submission "$submission" --backend RTX5090-SL3061 \
      --dataset-id suitesparse_sample_100 --operator "$operator" \
      --timeout 14000
  fi
}

configs=(
  coo-default coo-alg1 coo-alg2 csr-default csr-alg1 csr-alg2
  csc-default csc-alg1 csc-alg2
  sell-c1-default sell-c1-alg1 sell-c2-default sell-c2-alg1
  sell-c4-default sell-c4-alg1 sell-c8-default sell-c8-alg1
  sell-c16-default sell-c16-alg1 sell-c32-default sell-c32-alg1
  sell-c64-default sell-c64-alg1 sell-c128-default sell-c128-alg1
  sell-nrows-default sell-nrows-alg1
)

for dtype in fp32 fp64; do
  for config in "${configs[@]}"; do
    run submissions/spmv/cusparse spmv.csr."$dtype" "$config" || \
      echo "candidate failed: cuSPARSE $dtype $config" >&2
  done
done

for dtype in fp32 fp64; do
  for submission in csr5 csr_adaptive alphasparse ghost_sell; do
    run submissions/spmv/"$submission" spmv.csr."$dtype" || \
      echo "candidate failed: $submission $dtype" >&2
  done
done

find data/result_exports/rtx5090 -type f -name '*.csv' | sort

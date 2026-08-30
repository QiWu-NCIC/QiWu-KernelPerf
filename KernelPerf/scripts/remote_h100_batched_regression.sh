#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
export PYTHONPATH="/public/home/yuanjunkang/yjk/pydeps:/public/home/yuanjunkang/yjk/QiWu-KernelPerf/KernelPerf"
mkdir -p data
rm -rf data/result_exports/h100 data/source/h100
rm -f data/kernelperf-h100.sqlite

run() {
  local submission=$1 operator=$2 config_id=${3:-}
  local extra=()
  if [[ -n "$config_id" ]]; then extra+=(--configuration-id "$config_id"); fi
  echo "===== $submission $operator ${config_id:-all} ====="
  srun --account=ncic --gres=gpu:1 --cpus-per-task=32 --mem=64G --time=04:00:00 \
    --export=ALL bash -lc "cd '$HOME/yjk/QiWu-KernelPerf/KernelPerf'; export PYTHONPATH='/public/home/yuanjunkang/yjk/pydeps:/public/home/yuanjunkang/yjk/QiWu-KernelPerf/KernelPerf'; python3 -m kernelperf.cli evaluate --config config/service-h100.json --submission '$submission' --backend H100-SXM5-80GB --dataset-id suitesparse_sample_100 --operator '$operator' ${config_id:+--configuration-id '$config_id'} --timeout 14000" || echo "candidate failed: $submission $operator ${config_id:-all}" >&2
}

for dtype in fp32 fp64; do
  for config in coo-default coo-alg1 coo-alg2 csr-default csr-alg1 csr-alg2 csc-default csc-alg1 csc-alg2 sell-c1-default sell-c1-alg1 sell-c2-default sell-c2-alg1 sell-c4-default sell-c4-alg1 sell-c8-default sell-c8-alg1 sell-c16-default sell-c16-alg1 sell-c32-default sell-c32-alg1 sell-c64-default sell-c64-alg1 sell-c128-default sell-c128-alg1 sell-nrows-default sell-nrows-alg1; do
    run submissions/spmv/cusparse spmv.csr.$dtype "$config"
  done
done

for dtype in fp32 fp64; do
  run submissions/spmv/csr5 spmv.csr.$dtype
  run submissions/spmv/csr_adaptive spmv.csr.$dtype
  run submissions/spmv/alphasparse spmv.csr.$dtype
  run submissions/spmv/ghost_sell spmv.csr.$dtype
done

find data/result_exports/h100 -type f -name '*.csv' | sort

#!/usr/bin/env bash
set -euo pipefail

# H100 login nodes do not expose a GPU.  Keep the service and all submit
# clients inside one Slurm GPU allocation so the local worker executes on the
# allocated H100 instead of failing its nvidia-smi health check on login01.
cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
export PYTHONPATH="/public/home/yuanjunkang/yjk/pydeps:/public/home/yuanjunkang/yjk/QiWu-KernelPerf/KernelPerf"
SERVICE_CONFIG=${SERVICE_CONFIG:-config/service-h100.json}
export SERVICE_CONFIG
srun --account=ncic --gres=gpu:1 --cpus-per-task=32 --mem=64G --time=04:00:00 \
  bash -lc '
    set -euo pipefail
    cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
    rm -rf data/result_exports/h100 data/source/h100
    rm -f data/kernelperf-h100.sqlite
    mkdir -p data
    export PYTHONPATH="/public/home/yuanjunkang/yjk/pydeps:/public/home/yuanjunkang/yjk/QiWu-KernelPerf/KernelPerf"
    cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
    run() {
      echo "===== $* ====="
      if ! PYTHONPATH=. python3 -m kernelperf.cli evaluate --config "$SERVICE_CONFIG" \
        --backend H100-SXM5-80GB --dataset-id suitesparse_sample_100 "$@" \
        --timeout 14400; then
        echo "candidate group failed; retaining its partial CSV exports" >&2
      fi
    }
    run --submission submissions/spmv/cusparse --operator spmv.csr.fp32
    run --submission submissions/spmv/cusparse --operator spmv.csr.fp64
    run --submission submissions/spmv/csr5 --operator spmv.csr.fp32
    run --submission submissions/spmv/csr5 --operator spmv.csr.fp64
    run --submission submissions/spmv/csr_adaptive --operator spmv.csr.fp32
    run --submission submissions/spmv/csr_adaptive --operator spmv.csr.fp64
    run --submission submissions/spmv/alphasparse --operator spmv.csr.fp32
    run --submission submissions/spmv/alphasparse --operator spmv.csr.fp64
    run --submission submissions/spmv/ghost_sell --operator spmv.csr.fp32
    run --submission submissions/spmv/ghost_sell --operator spmv.csr.fp64
    find "$HOME/yjk/QiWu-KernelPerf/KernelPerf/data/result_exports" -type f -name "*.csv" | sort
  '

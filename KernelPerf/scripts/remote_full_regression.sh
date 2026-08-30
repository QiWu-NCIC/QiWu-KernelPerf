#!/usr/bin/env bash
set -u

PLATFORM=${1:?platform name}
CONFIG=${2:?service config}
cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"
set -e
rm -rf "data/result_exports/${PLATFORM}" "data/source/${PLATFORM}"
rm -f data/kernelperf-*.sqlite
mkdir -p data
cd "$HOME/yjk/QiWu-KernelPerf/KernelPerf"

run() {
  echo "===== $* ====="
  if ! PYTHONPATH=. python3 -m kernelperf.cli evaluate --config "$CONFIG" --backend "$PLATFORM" --dataset-id suitesparse_sample_100 "$@" --timeout 14400; then
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

echo "completed ${PLATFORM}; exports:"
find "$HOME/yjk/QiWu-KernelPerf/KernelPerf/data/result_exports/${PLATFORM}" -type f -name '*.csv' | sort

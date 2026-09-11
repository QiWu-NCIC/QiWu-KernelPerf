#!/usr/bin/env bash
set -euo pipefail

# Install the third-party CUDA SpMV library exposed to submitted adapters.
# The resulting manifest is consumed during worker audits; submissions never
# upload this source tree or choose its compiler/link flags.
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
PREFIX_GHOST=${PREFIX_GHOST:-${HOME}/.local/ghost-cuda}
SRC_ROOT=${SRC_ROOT:-${HOME}/.cache/kernelperf-spmv-src}
DEPENDENCY_MANIFEST=${DEPENDENCY_MANIFEST:-${HOME}/.config/kernelperf/spmv-dependencies.json}
GHOST_COMMIT=${GHOST_COMMIT:-22a004dbbfb604d7b04b0bde813c8100d403cfcc}
CUDA_ARCH=${CUDA_ARCH:-80}

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA toolkit not found at $CUDA_HOME; set CUDA_HOME to the pinned worker toolkit" >&2
  exit 2
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${SKIP_APT:-0}" != "1" && "${EUID:-$(id -u)}" == "0" ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git libhwloc-dev libopenblas-dev
fi
mkdir -p "$SRC_ROOT"

clone_at() {
  local url=$1 dir=$2 commit=$3
  if [[ ! -d "$dir/.git" ]]; then git clone --no-checkout "$url" "$dir"; fi
  git -C "$dir" fetch --tags --force origin
  git -C "$dir" checkout --detach "$commit"
}

clone_at https://github.com/RRZE-HPC/GHOST.git "$SRC_ROOT/GHOST" "$GHOST_COMMIT"
# The pinned upstream CMake file still emits Kepler code (compute_35), which
# CUDA 12.x no longer accepts. Keep the source pin and replace only that build
# flag for the A100 worker.
sed -i 's/-gencode arch=compute_35,code=sm_35/-gencode arch=compute_80,code=sm_80/' "$SRC_ROOT/GHOST/CMakeLists.txt"
# CUDA 12 removed the legacy cuSPARSE csrmv/csrmm APIs used only by GHOST's
# C=1 compatibility path. KernelPerf fixes C=32, but libghost still exports
# the C=1 selector, so keep that unused ABI as an explicit unsupported stub.
install -m 0644 "$(dirname "$0")/ghost_sell1_cuda12_compat.cu" \
  "$SRC_ROOT/GHOST/src/sell-1_spmv.cu"
rm -rf /tmp/kernelperf-GHOST-build
BUILD_ROOT=${BUILD_ROOT:-${TMPDIR:-/tmp}/kernelperf-GHOST-build}
cmake -S "$SRC_ROOT/GHOST" -B "$BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release -DGHOST_USE_MPI=OFF -DGHOST_USE_CUDA=ON \
  -DGHOST_USE_SPMP=OFF -DGHOST_USE_OPENMP=OFF \
  -DGHOST_IDX64_GLOBAL=OFF -DGHOST_IDX64_LOCAL=OFF \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_INSTALL_PREFIX="$PREFIX_GHOST"
cmake --build "$BUILD_ROOT" -j"$(nproc)"
cmake --install "$BUILD_ROOT"

# GHOST's pinned hwloc integration references an API removed from newer hwloc.
# The MIC lookup is irrelevant on NVIDIA workers; provide the missing no-op ABI
# so the same pinned GHOST binary links on current Ubuntu distributions.
mkdir -p "$PREFIX_GHOST/lib/compat"
cat >"$BUILD_ROOT/hwloc_compat.c" <<'EOF'
#include <stddef.h>
void *hwloc_intel_mic_get_device_osdev_by_index(void *topology, unsigned index)
{
    (void)topology;
    (void)index;
    return NULL;
}
EOF
cc -shared -fPIC -O2 "$BUILD_ROOT/hwloc_compat.c" \
  -o "$PREFIX_GHOST/lib/compat/libhwloc-compat.so"

mkdir -p "$(dirname "$DEPENDENCY_MANIFEST")"
cat > "$DEPENDENCY_MANIFEST" <<EOF
{
  "cuda": "$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1)",
  "GHOST": {"commit": "$GHOST_COMMIT", "prefix": "$PREFIX_GHOST", "C": 32, "cuda12_sell1_compat": true, "sigma_candidates": [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]},
  "indices": {"global_bits": 32, "local_bits": 32}
}
EOF
echo "installed $DEPENDENCY_MANIFEST"

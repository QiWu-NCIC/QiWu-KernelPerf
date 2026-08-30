#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/yjk"
rm -rf ghost-src ghost-build
tar -xzf GHOST-qiwu.tgz
mv GHOST-qiwu ghost-src
rm -rf hwloc-include
mkdir -p hwloc-include
if [[ -f /usr/include/hwloc.h ]]; then
  ln -s /usr/include/hwloc.h hwloc-include/hwloc.h
  [[ -d /usr/include/hwloc ]] && cp -a /usr/include/hwloc/. hwloc-include/hwloc/
else
  HPCX_HWLOC=/public/software/hpcx/2.20/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include
  cp -a "$HPCX_HWLOC/." hwloc-include/
  # OpenMPI's vendored headers are symbol-renamed; the system runtime uses
  # the normal hwloc ABI, so disable that private prefix for this build.
  if [[ -f hwloc-include/hwloc/autogen/config.h ]]; then
    sed -i 's/#define HWLOC_SYM_TRANSFORM 1/#define HWLOC_SYM_TRANSFORM 0/' \
      hwloc-include/hwloc/autogen/config.h
  fi
fi
if [[ ! -f hwloc-include/hwloc/intel-mic.h ]]; then
  : > hwloc-include/hwloc/intel-mic.h
fi
if [[ ! -f /usr/include/cblas.h && ! -f /usr/include/x86_64-linux-gnu/cblas.h ]]; then
  cp cblas_stub.h ghost-src/include/cblas.h
fi
if [[ ! -f /usr/include/cblas.h && ! -f /usr/include/x86_64-linux-gnu/cblas.h ]]; then
  # Minimal user-space CBLAS shim for clusters without a BLAS module.  GHOST's
  # SELL SpMV path does not use GEMM, but providing the symbols keeps the
  # optional BLAS objects linkable.
  cat > cblas_stub.c <<'EOF'
#include <stddef.h>
void cblas_sgemm(int o,int ta,int tb,int m,int n,int k,float a,const float*A,int lda,const float*B,int ldb,float b,float*C,int ldc){(void)o;(void)ta;(void)tb;(void)m;(void)n;(void)k;(void)a;(void)A;(void)lda;(void)B;(void)ldb;(void)b;(void)C;(void)ldc;}
void cblas_dgemm(int o,int ta,int tb,int m,int n,int k,double a,const double*A,int lda,const double*B,int ldb,double b,double*C,int ldc){(void)o;(void)ta;(void)tb;(void)m;(void)n;(void)k;(void)a;(void)A;(void)lda;(void)B;(void)ldb;(void)b;(void)C;(void)ldc;}
void cblas_cgemm(int o,int ta,int tb,int m,int n,int k,const void*a,const void*A,int lda,const void*B,int ldb,const void*b,void*C,int ldc){(void)o;(void)ta;(void)tb;(void)m;(void)n;(void)k;(void)a;(void)A;(void)lda;(void)B;(void)ldb;(void)b;(void)C;(void)ldc;}
void cblas_zgemm(int o,int ta,int tb,int m,int n,int k,const void*a,const void*A,int lda,const void*B,int ldb,const void*b,void*C,int ldc){(void)o;(void)ta;(void)tb;(void)m;(void)n;(void)k;(void)a;(void)A;(void)lda;(void)B;(void)ldb;(void)b;(void)C;(void)ldc;}
EOF
  cc -fPIC -shared cblas_stub.c -o libcblas.so
fi
cp KernelPerf/scripts/ghost_sell1_cuda12_compat.cu ghost-src/src/sell-1_spmv.cu
# Keep the source portable; the deployment host selects the CUDA architecture
# through CMAKE_CUDA_ARCHITECTURES below (SM 90 on H100, SM 120 on RTX 5090).
sed -i \
  -e 's/compute_35,code=sm_35/compute_90,code=sm_90/g' \
  -e 's/compute_70,code=sm_70/compute_90,code=sm_90/g' \
  ghost-src/CMakeLists.txt
python3 - <<'PY'
from pathlib import Path

header = Path("ghost-src/include/ghost/cu_sell_kernel.h")
text = header.read_text()
needle = "template<typename v_t>\n__device__ inline v_t ghost_shfl_down32"
declaration = (
    "template<typename v_t>\n"
    "__device__ inline v_t ghost_shfl_down(v_t var, unsigned int srcLane, int width);\n\n"
)
if declaration not in text:
    if needle not in text:
        raise SystemExit("unexpected GHOST CUDA shuffle header")
    text = text.replace(needle, declaration + needle, 1)
    header.write_text(text)

machine = Path("ghost-src/src/machine.c")
text = machine.read_text()
needle = '#include "ghost/machine.h"'
compat = (
    "#ifndef HWLOC_TOPOLOGY_FLAG_IO_DEVICES\n"
    "#define HWLOC_TOPOLOGY_FLAG_IO_DEVICES 0\n"
    "#endif\n"
    "#ifndef HWLOC_OBJ_CACHE\n"
    "#define HWLOC_OBJ_CACHE HWLOC_OBJ_L1CACHE\n"
    "#endif\n"
)
if compat not in text:
    if needle not in text:
        raise SystemExit("unexpected GHOST machine source")
    text = text.replace(needle, needle + "\n" + compat, 1)
    machine.write_text(text)
PY
GPU_ARCH=${GPU_ARCH:-120}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
if [[ -x /public/software/cmake/3.31.12/bin/cmake ]]; then
  export PATH=/public/software/cmake/3.31.12/bin:$PATH
fi
# GHOST forwards HWLOC_INCLUDE_DIR to nvcc as -isystem.  The multiarch
# include directory avoids shadowing nvcc's standard headers.
if [[ -f /usr/lib/x86_64-linux-gnu/libblas.so ]]; then
  BLAS_LIBRARIES=${BLAS_LIBRARIES:-/usr/lib/x86_64-linux-gnu/libblas.so}
else
  BLAS_LIBRARIES=${BLAS_LIBRARIES:-$HOME/yjk/libcblas.so}
fi
cmake -S ghost-src -B ghost-build \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_BUILD_TYPE=Release -DGHOST_BUILD_TEST=OFF -DGHOST_USE_MPI=OFF \
  -DGHOST_USE_CUDA=ON -DGHOST_USE_SPMP=OFF -DGHOST_USE_OPENMP=OFF \
  -DGHOST_IDX64_GLOBAL=OFF -DGHOST_IDX64_LOCAL=OFF \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/ghost-cuda" \
  -DHWLOC_INCLUDE_DIR="$HOME/yjk/hwloc-include" \
  -DLIBHWLOC=/lib/x86_64-linux-gnu/libhwloc.so.15 \
  -DCBLAS_INCLUDE_DIR="${CBLAS_INCLUDE_DIR:-$HOME/yjk/ghost-src/include}" \
  -DBLAS_LIBRARIES="$BLAS_LIBRARIES"
cmake --build ghost-build -j8
cmake --install ghost-build


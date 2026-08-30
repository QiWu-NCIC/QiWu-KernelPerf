#include <ghost.h>

// GHOST's C=1 path uses legacy cuSPARSE APIs removed by CUDA 12. KernelPerf's
// SELL-C-sigma candidates use C=32, so preserve the unused ABI explicitly.
extern "C" ghost_error ghost_cu_sell1_spmv_selector(
    ghost_densemat*, ghost_sparsemat*, ghost_densemat*, ghost_spmv_opts) {
    return GHOST_ERR_NOT_IMPLEMENTED;
}

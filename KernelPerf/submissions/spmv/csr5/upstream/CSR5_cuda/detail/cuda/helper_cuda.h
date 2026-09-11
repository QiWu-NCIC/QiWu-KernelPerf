#ifndef CSR5_LOCAL_HELPER_CUDA_H
#define CSR5_LOCAL_HELPER_CUDA_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void csr5_check_cuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                     file, line, static_cast<int>(result), cudaGetErrorString(result), func);
        std::exit(EXIT_FAILURE);
    }
}

#ifndef checkCudaErrors
#define checkCudaErrors(val) csr5_check_cuda((val), #val, __FILE__, __LINE__)
#endif

#endif

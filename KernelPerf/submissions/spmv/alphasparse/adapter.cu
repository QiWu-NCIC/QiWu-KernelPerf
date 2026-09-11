#include <qiwu/spmv_plugin.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// Load the public status and scalar definitions without the incomplete CUDA
// FP16 declarations in the upstream umbrella header. The selected FP32/FP64
// kernels only require a handle carrying a CUDA stream.
#ifdef __CUDA__
#undef __CUDA__
#endif
#define device alpha_device
#define __HYGON__ 1
#include "upstream/include/alphasparse.h"
#undef __HYGON__
#undef device
#undef cuFloatComplex
#undef cuDoubleComplex

#include <cuComplex.h>

struct QiwuAlphaSparseHandle {
    cudaStream_t stream = nullptr;
    int wavefront_size = 32;
};

#define alphasparseHandle_t QiwuAlphaSparseHandle*
#define __CUDA__ 1
#if defined(_WIN32)
#define __attribute__(...)
#endif

// The complete AlphaSparse/Library commit 39734b2 is vendored under upstream/.
// These are the CUDA kernels selected by its alphasparseSpMV CSR dispatch.
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include "upstream/cuda/kernel/level2/alphasparse_spmv_csr_scalar.h"
#include "upstream/cuda/kernel/level2/alphasparse_spmv_csr_vector.h"
#include "upstream/cuda/kernel/level2/alphasparse_spmv_csr_merge_ginkgo.h"
#include "upstream/cuda/kernel/level2/alphasparse_spmv_csr_line_enhance.h"
#include "upstream/cuda/kernel/level2/alphasparse_spmv_csr_flat.h"

#undef alphasparseHandle_t
#if defined(_WIN32)
#undef __attribute__
#endif

enum class AlphaSparseAlgorithm : int {
    scalar = 1,
    vector = 2,
    merge = 4,
    line_enhance = 5,
    flat1 = 6,
    flat4 = 7,
    flat8 = 8,
};

#ifndef KERNELPERF_ALPHASPARSE_ALGORITHM
#define KERNELPERF_ALPHASPARSE_ALGORITHM 1
#endif

struct QiwuSpmvStorage {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    const int32_t* row_offsets = nullptr;
    const int32_t* column_indices = nullptr;
    const QiwuSpmvScalar* values = nullptr;
    const QiwuSpmvScalar* x = nullptr;
    QiwuSpmvScalar* y = nullptr;
    void* workspace = nullptr;
    QiwuAlphaSparseHandle handle{};
};

namespace {

inline void check(cudaError_t status, const char* operation) {
    qiwu_spmv_check_cuda(status, operation);
}

inline size_t merge_workspace(int64_t rows, int64_t nnz) {
    constexpr int64_t items_per_block = 512 * 8;
    const int64_t blocks = (rows + nnz + items_per_block - 1) / items_per_block;
    return static_cast<size_t>(2 * (std::max<int64_t>(blocks, 1) + 1)) * sizeof(int32_t);
}

inline size_t flat_workspace(int64_t nnz) {
    const int64_t partitions = (nnz + 2 * 512 - 1) / (2 * 512);
    return static_cast<size_t>(std::max<int64_t>(partitions, 1) + 1) * sizeof(int32_t);
}

template <typename T>
inline alphasparseStatus_t solve_with_upstream(
    QiwuSpmvStorage* storage, AlphaSparseAlgorithm algorithm, T alpha, T beta
) {
    auto* handle = &storage->handle;
    const T* values = storage->values;
    const int32_t* row_offsets = storage->row_offsets;
    const int32_t* columns = storage->column_indices;
    const T* x = storage->x;
    T* y = storage->y;
    const int32_t m = static_cast<int32_t>(storage->rows);
    const int32_t n = static_cast<int32_t>(storage->cols);
    const int32_t nnz = static_cast<int32_t>(storage->nnz);

    switch (algorithm) {
    case AlphaSparseAlgorithm::scalar:
        return spmv_csr_scalar<int32_t, T, T, T>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y);
    case AlphaSparseAlgorithm::vector:
        return spmv_csr_vector<int32_t, T, T, T>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y);
    case AlphaSparseAlgorithm::merge:
        return spmv_csr_merge_ginkgo<int32_t, T, T, T>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y,
            storage->workspace);
    case AlphaSparseAlgorithm::line_enhance:
        return spmv_csr_line_adaptive<int32_t, T, T, T>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y,
            storage->workspace);
    case AlphaSparseAlgorithm::flat1:
        return spmv_csr_flat<int32_t, T, T, T, 1>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y,
            storage->workspace);
    case AlphaSparseAlgorithm::flat4:
        return spmv_csr_flat<int32_t, T, T, T, 4>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y,
            storage->workspace);
    case AlphaSparseAlgorithm::flat8:
        return spmv_csr_flat<int32_t, T, T, T, 8>(
            handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y,
            storage->workspace);
    }
    throw std::runtime_error("unknown AlphaSparseLib algorithm selector");
}

inline void clear_output(QiwuSpmvStorage* storage, cudaStream_t stream) {
    check(cudaMemsetAsync(
        storage->y, 0,
        static_cast<size_t>(storage->rows) * sizeof(QiwuSpmvScalar), stream
    ), "clear AlphaSparse output");
}

}  // namespace

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz < 0 ||
        input->rows > std::numeric_limits<int32_t>::max() ||
        input->cols > std::numeric_limits<int32_t>::max() ||
        input->nnz > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("unsupported AlphaSparse CSR input");
    }
    auto* storage = new QiwuSpmvStorage();
    storage->rows = input->rows;
    storage->cols = input->cols;
    storage->nnz = input->nnz;
    storage->row_offsets = input->device_row_offsets;
    storage->column_indices = input->device_column_indices;
    storage->values = input->device_values;
    storage->x = context->device_x;
    storage->y = context->device_y;
    storage->handle.stream = stream;
    storage->handle.wavefront_size = WARP_SIZE;

    constexpr auto algorithm = static_cast<AlphaSparseAlgorithm>(KERNELPERF_ALPHASPARSE_ALGORITHM);
    size_t workspace_bytes = 0;
    if (algorithm == AlphaSparseAlgorithm::merge) {
        workspace_bytes = merge_workspace(storage->rows, storage->nnz);
    } else if (algorithm == AlphaSparseAlgorithm::flat1
            || algorithm == AlphaSparseAlgorithm::flat4
            || algorithm == AlphaSparseAlgorithm::flat8) {
        workspace_bytes = flat_workspace(storage->nnz);
    }
    if (workspace_bytes != 0) {
        check(cudaMalloc(&storage->workspace, workspace_bytes),
              "allocate AlphaSparseLib workspace");
    }
    // Output initialization is part of setup, so it is excluded from the
    // solve-only event timing. The benchmark requests another reset only for
    // its out-of-band validation call.
    clear_output(storage, stream);
    return storage;
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false) {
    if (!storage) throw std::runtime_error("AlphaSparseLib storage is null");
    if (context && context->reset_output) {
        clear_output(storage, stream);
    }
    storage->handle.stream = stream;
    const auto algorithm = static_cast<AlphaSparseAlgorithm>(KERNELPERF_ALPHASPARSE_ALGORITHM);
    const auto status = solve_with_upstream<QiwuSpmvScalar>(
        storage, algorithm, QiwuSpmvScalar{1}, QiwuSpmvScalar{0});
    if (status != ALPHA_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("AlphaSparse SpMV returned an error");
    }
    check(cudaGetLastError(), "launch AlphaSparse SpMV");
}

extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage* storage, cudaStream_t) noexcept(false) {
    if (!storage) return;
    if (storage->workspace) check(cudaFree(storage->workspace), "free AlphaSparse workspace");
    delete storage;
}

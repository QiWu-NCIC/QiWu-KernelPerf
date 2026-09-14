#include <qiwu/spmv_plugin.cuh>

#if defined(QIWU_BACKEND_HIP)

// The upstream AlphaSparse tree ships a native HIP implementation.  Keep the
// adapter deliberately thin: the benchmark only owns the lifecycle glue and
// calls the same CSR kernels selected by the upstream dispatch.
#define __HIP__ 1
#define __HIPNEED__ 1
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#include "upstream/include/alphasparse.h"
#include "upstream/hip/kernel/level2/alphasparse_spmv_csr_scalar.h"
#include "upstream/hip/kernel/level2/alphasparse_spmv_csr_vector.h"
#include "upstream/hip/kernel/level2/alphasparse_spmv_csr_merge_ginkgo.h"
#include "upstream/hip/kernel/level2/alphasparse_spmv_csr_line_enhance.h"
#include "upstream/hip/kernel/level2/alphasparse_spmv_csr_flat.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

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
    alphasparse_handle handle{};
};

namespace {

inline void check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
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
alphasparseStatus_t merge_on_stream(QiwuSpmvStorage* storage, T alpha, T beta, hipStream_t stream) {
    constexpr int block_size = 512;
    constexpr int items_per_thread = 8;
    constexpr int items_per_block = block_size * items_per_thread;
    const int32_t rows = static_cast<int32_t>(storage->rows);
    const int32_t nnz = static_cast<int32_t>(storage->nnz);
    const int32_t merge_items = rows + nnz;
    const int32_t blocks = ceildivT(merge_items, items_per_block);
    auto* block_start_x = static_cast<int32_t*>(storage->workspace);
    auto* block_start_y = block_start_x + blocks + 1;
    const int shared_bytes = items_per_block * (sizeof(T) + sizeof(int32_t));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(abstract_merge_path_search<int32_t>),
        ceildivT(blocks + rows + 1, block_size), block_size, 0, stream,
        block_start_x, block_start_y, rows, nnz, merge_items,
        items_per_block, blocks, storage->row_offsets + 1, storage->y, beta);
    if (blocks > 0) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(merge_path_spmv<items_per_thread>),
            blocks, block_size, shared_bytes, stream,
            rows, nnz, merge_items, alpha, storage->values,
            storage->column_indices, storage->row_offsets + 1, storage->x,
            beta, storage->y, block_start_x, block_start_y);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define QIWU_LAUNCH_LINE(REDUCE, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS) \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(line_enhance_kernel<REDUCE, __WF_SIZE__, VEC_SIZE,     \
            ROWS_PER_BLOCK, R, THREADS, int32_t, T, T, T>),                   \
        BLOCKS, THREADS, 0, stream, rows, alpha, beta, storage->row_offsets,  \
        storage->column_indices, storage->values, storage->x, storage->y)

template <typename T>
alphasparseStatus_t line_enhance_on_stream(
    QiwuSpmvStorage* storage, T alpha, T beta, hipStream_t stream) {
    constexpr int threads = 512;
    const int32_t rows = static_cast<int32_t>(storage->rows);
    const int32_t nnz = static_cast<int32_t>(storage->nnz);
    const int32_t average_row_nnz = nnz / rows;
    if (nnz <= (1 << 24)) {
        if (average_row_nnz >= 32) {
            const int blocks = ceildivT(rows, 64);
            QIWU_LAUNCH_LINE(LE_REDUCE_OPTION_VEC, 64, 8, 4, blocks, threads);
        } else {
            const int blocks = ceildivT(rows, 64);
            QIWU_LAUNCH_LINE(LE_REDUCE_OPTION_DIRECT, 64, 1, 2, blocks, threads);
        }
    } else if (average_row_nnz >= 24) {
        const int blocks = ceildivT(rows, 64);
        QIWU_LAUNCH_LINE(LE_REDUCE_OPTION_VEC, 64, 4, 2, blocks, threads);
    } else {
        const int blocks = ceildivT(rows, 128);
        QIWU_LAUNCH_LINE(LE_REDUCE_OPTION_DIRECT, 128, 1, 2, blocks, threads);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#undef QIWU_LAUNCH_LINE

template <typename T, int vector_size>
alphasparseStatus_t flat_on_stream(
    QiwuSpmvStorage* storage, T alpha, T beta, hipStream_t stream) {
    constexpr int32_t block_size = 512;
    constexpr int32_t nnz_per_block = 2 * block_size;
    const int32_t rows = static_cast<int32_t>(storage->rows);
    const int32_t nnz = static_cast<int32_t>(storage->nnz);
    const int32_t blocks = ceildivT(nnz, nnz_per_block);
    const int32_t partition_blocks = ceildivT(blocks + rows, block_size);
    auto* partition = static_cast<int32_t*>(storage->workspace);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(balanced_partition_row_by_nnz_flat_and_scale_y<int32_t, T, T>),
        partition_blocks, block_size, 0, stream,
        storage->row_offsets, rows, blocks, partition, nnz, nnz_per_block,
        beta, storage->y);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(spmv_flat<int32_t, T, T, T, block_size, nnz_per_block, vector_size>),
        blocks, block_size, 0, stream,
        rows, nnz, alpha, partition, storage->row_offsets,
        storage->column_indices, storage->values, storage->x, beta, storage->y);
    return ALPHA_SPARSE_STATUS_SUCCESS;
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
        return spmv_csr_scalar<int32_t, T, T, T>(handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y);
    case AlphaSparseAlgorithm::vector:
        return spmv_csr_vector<int32_t, T, T, T>(handle, m, n, nnz, alpha, values, row_offsets, columns, x, beta, y);
    case AlphaSparseAlgorithm::merge:
        return merge_on_stream(storage, alpha, beta, handle->stream);
    case AlphaSparseAlgorithm::line_enhance:
        return line_enhance_on_stream(storage, alpha, beta, handle->stream);
    case AlphaSparseAlgorithm::flat1:
        return flat_on_stream<T, 1>(storage, alpha, beta, handle->stream);
    case AlphaSparseAlgorithm::flat4:
        return flat_on_stream<T, 4>(storage, alpha, beta, handle->stream);
    case AlphaSparseAlgorithm::flat8:
        return flat_on_stream<T, 8>(storage, alpha, beta, handle->stream);
    }
    throw std::runtime_error("unknown AlphaSparseLib algorithm selector");
}

inline void clear_output(QiwuSpmvStorage* storage, hipStream_t stream) {
    check(hipMemsetAsync(storage->y, 0, static_cast<size_t>(storage->rows) * sizeof(QiwuSpmvScalar), stream), "clear AlphaSparse output");
}

}  // namespace

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    hipStream_t stream
) noexcept(false) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE || input->rows <= 0 || input->cols <= 0 || input->nnz < 0 ||
        input->rows > std::numeric_limits<int32_t>::max() || input->cols > std::numeric_limits<int32_t>::max() || input->nnz > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("unsupported AlphaSparse HIP CSR input");
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
    int device = 0;
    check(hipGetDevice(&device), "get HIP device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, device), "get HIP device properties");
    storage->handle.device = device;
    storage->handle.properties = properties;
    storage->handle.wavefront_size = properties.warpSize;

    constexpr auto algorithm = static_cast<AlphaSparseAlgorithm>(KERNELPERF_ALPHASPARSE_ALGORITHM);
    size_t workspace_bytes = 0;
    if (algorithm == AlphaSparseAlgorithm::merge) workspace_bytes = merge_workspace(storage->rows, storage->nnz);
    if (algorithm == AlphaSparseAlgorithm::flat1 || algorithm == AlphaSparseAlgorithm::flat4 || algorithm == AlphaSparseAlgorithm::flat8) workspace_bytes = flat_workspace(storage->nnz);
    if (workspace_bytes != 0) check(hipMalloc(&storage->workspace, workspace_bytes), "allocate AlphaSparse HIP workspace");
    clear_output(storage, stream);
    return storage;
}

extern "C" void qiwu_spmv_solve(QiwuSpmvStorage* storage, const QiwuSpmvExecutionContext* context, hipStream_t stream) noexcept(false) {
    if (!storage) throw std::runtime_error("AlphaSparse HIP storage is null");
    if (context && context->reset_output) clear_output(storage, stream);
    storage->handle.stream = stream;
    const auto status = solve_with_upstream<QiwuSpmvScalar>(storage, static_cast<AlphaSparseAlgorithm>(KERNELPERF_ALPHASPARSE_ALGORITHM), QiwuSpmvScalar{1}, QiwuSpmvScalar{0});
    if (status != ALPHA_SPARSE_STATUS_SUCCESS) throw std::runtime_error("AlphaSparse HIP SpMV returned an error");
    check(hipGetLastError(), "launch AlphaSparse HIP SpMV");
}

extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage* storage, hipStream_t) noexcept(false) {
    if (!storage) return;
    if (storage->workspace) check(hipFree(storage->workspace), "free AlphaSparse HIP workspace");
    delete storage;
}

#else

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

#endif  // QIWU_BACKEND_HIP

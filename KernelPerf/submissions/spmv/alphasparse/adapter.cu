#include <qiwu/spmv_plugin.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef KERNELPERF_ALPHASPARSE_ALGORITHM
#define KERNELPERF_ALPHASPARSE_ALGORITHM 1
#endif

// CUDA port of the CSR algorithm families in alphasparse_for_test/hip/kernel/level2:
// scalar, vector, merge, line-enhance and flat (unroll 1/4/8). The submission
// keeps only this adapter and candidate selectors; the full AlphaSparseLib tree is
// not uploaded to a worker.

struct QiwuSpmvStorage {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    const int32_t* row_offsets = nullptr;
    const int32_t* column_indices = nullptr;
    const QiwuSpmvScalar* values = nullptr;
    const QiwuSpmvScalar* x = nullptr;
    QiwuSpmvScalar* y = nullptr;
    QiwuSpmvScalar* merge_partials = nullptr;
};

namespace kernelperf_alphasparse {

inline void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <int UNROLL>
__global__ void scalar_kernel(
    int64_t rows, const int32_t* row_offsets, const int32_t* columns,
    const QiwuSpmvScalar* values, const QiwuSpmvScalar* x, QiwuSpmvScalar* y) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int32_t begin = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    QiwuSpmvScalar sum = 0;
    int32_t index = begin;
    for (; index + UNROLL <= end; index += UNROLL) {
#pragma unroll
        for (int lane = 0; lane < UNROLL; ++lane) {
            sum += values[index + lane] * x[columns[index + lane]];
        }
    }
    for (; index < end; ++index) sum += values[index] * x[columns[index]];
    y[row] = sum;
}

__global__ void vector_kernel(
    int64_t rows, const int32_t* row_offsets, const int32_t* columns,
    const QiwuSpmvScalar* values, const QiwuSpmvScalar* x, QiwuSpmvScalar* y) {
    const int lane = threadIdx.x & 31;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    QiwuSpmvScalar sum = 0;
    for (int32_t index = row_offsets[row] + lane; index < row_offsets[row + 1]; index += 32) {
        sum += values[index] * x[columns[index]];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void merge_products(
    int64_t nnz, const int32_t* columns, const QiwuSpmvScalar* values,
    const QiwuSpmvScalar* x, QiwuSpmvScalar* partials) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < nnz) partials[index] = values[index] * x[columns[index]];
}

__global__ void merge_reduce_rows(
    int64_t rows, const int32_t* row_offsets, const QiwuSpmvScalar* partials, QiwuSpmvScalar* y) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    QiwuSpmvScalar sum = 0;
    for (int32_t index = row_offsets[row]; index < row_offsets[row + 1]; ++index) sum += partials[index];
    y[row] = sum;
}

template <int ROWS_PER_BLOCK>
__global__ void line_enhance_kernel(
    int64_t rows, const int32_t* row_offsets, const int32_t* columns,
    const QiwuSpmvScalar* values, const QiwuSpmvScalar* x, QiwuSpmvScalar* y) {
    __shared__ QiwuSpmvScalar row_sums[ROWS_PER_BLOCK];
    const int local = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK + local;
    QiwuSpmvScalar sum = 0;
    if (row < rows) {
        for (int32_t index = row_offsets[row]; index < row_offsets[row + 1]; ++index)
            sum += values[index] * x[columns[index]];
    }
    row_sums[local] = sum;
    __syncthreads();
    if (row < rows) y[row] = row_sums[local];
}

template <int UNROLL>
inline void launch_scalar(QiwuSpmvStorage* storage, cudaStream_t stream) {
    constexpr int block = 256;
    const int grid = static_cast<int>((storage->rows + block - 1) / block);
    scalar_kernel<UNROLL><<<grid, block, 0, stream>>>(
        storage->rows, storage->row_offsets, storage->column_indices,
        storage->values, storage->x, storage->y);
}

inline void launch_vector(QiwuSpmvStorage* storage, cudaStream_t stream) {
    constexpr int block = 256;
    const int grid = static_cast<int>((storage->rows * 32 + block - 1) / block);
    vector_kernel<<<grid, block, 0, stream>>>(
        storage->rows, storage->row_offsets, storage->column_indices,
        storage->values, storage->x, storage->y);
}

inline void launch_merge(QiwuSpmvStorage* storage, cudaStream_t stream) {
    constexpr int block = 256;
    const int product_grid = static_cast<int>((storage->nnz + block - 1) / block);
    const int row_grid = static_cast<int>((storage->rows + block - 1) / block);
    merge_products<<<product_grid, block, 0, stream>>>(
        storage->nnz, storage->column_indices, storage->values, storage->x,
        storage->merge_partials);
    merge_reduce_rows<<<row_grid, block, 0, stream>>>(
        storage->rows, storage->row_offsets, storage->merge_partials, storage->y);
}

inline void launch_line(QiwuSpmvStorage* storage, cudaStream_t stream) {
    constexpr int rows_per_block = 32;
    const int grid = static_cast<int>((storage->rows + rows_per_block - 1) / rows_per_block);
    line_enhance_kernel<rows_per_block><<<grid, rows_per_block, 0, stream>>>(
        storage->rows, storage->row_offsets, storage->column_indices,
        storage->values, storage->x, storage->y);
}

}  // namespace kernelperf_alphasparse

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input, const QiwuSpmvExecutionContext* context, cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_alphasparse;
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE)
        throw std::runtime_error("invalid AlphaSparseLib CSR input");
    auto* storage = new QiwuSpmvStorage();
    storage->rows = input->rows;
    storage->cols = input->cols;
    storage->nnz = input->nnz;
    storage->row_offsets = input->device_row_offsets;
    storage->column_indices = input->device_column_indices;
    storage->values = input->device_values;
    storage->x = context->device_x;
    storage->y = context->device_y;
    if (KERNELPERF_ALPHASPARSE_ALGORITHM == 3) {
        check(cudaMalloc(reinterpret_cast<void**>(&storage->merge_partials),
                         static_cast<size_t>(storage->nnz) * sizeof(QiwuSpmvScalar)),
              "allocate AlphaSparseLib merge workspace");
    }
    check(cudaMemsetAsync(storage->y, 0, static_cast<size_t>(storage->rows) * sizeof(QiwuSpmvScalar), stream),
          "initialize AlphaSparseLib output");
    return storage;
}

extern "C" void qiwu_spmv_solve(QiwuSpmvStorage* storage,
                                           const QiwuSpmvExecutionContext*, cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_alphasparse;
    if (!storage) throw std::runtime_error("AlphaSparseLib storage is null");
    if (KERNELPERF_ALPHASPARSE_ALGORITHM == 1) launch_scalar<1>(storage, stream);
    else if (KERNELPERF_ALPHASPARSE_ALGORITHM == 2) launch_vector(storage, stream);
    else if (KERNELPERF_ALPHASPARSE_ALGORITHM == 3) launch_merge(storage, stream);
    else if (KERNELPERF_ALPHASPARSE_ALGORITHM == 4) launch_line(storage, stream);
    else if (KERNELPERF_ALPHASPARSE_ALGORITHM == 5) launch_scalar<1>(storage, stream);
    else if (KERNELPERF_ALPHASPARSE_ALGORITHM == 6) launch_scalar<4>(storage, stream);
    else if (KERNELPERF_ALPHASPARSE_ALGORITHM == 7) launch_scalar<8>(storage, stream);
    else throw std::runtime_error("unknown AlphaSparseLib algorithm selector");
    check(cudaGetLastError(), "launch AlphaSparseLib SpMV");
}

extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage* storage, cudaStream_t) noexcept(false) {
    if (!storage) return;
    if (storage->merge_partials) cudaFree(storage->merge_partials);
    delete storage;
}

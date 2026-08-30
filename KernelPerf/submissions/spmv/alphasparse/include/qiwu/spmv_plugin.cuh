#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

enum class QiwuSpmvDataType : uint32_t {
    fp32 = 1,
    fp64 = 2,
};

#if defined(QIWU_SPMV_FP64)
using QiwuSpmvScalar = double;
constexpr QiwuSpmvDataType QIWU_SPMV_DATA_TYPE = QiwuSpmvDataType::fp64;
#else
using QiwuSpmvScalar = float;
constexpr QiwuSpmvDataType QIWU_SPMV_DATA_TYPE = QiwuSpmvDataType::fp32;
#endif

struct QiwuSpmvCsrInput {
    QiwuSpmvDataType data_type;
    int64_t rows;
    int64_t cols;
    int64_t nnz;

    const int32_t* host_row_offsets;
    const int32_t* host_column_indices;
    const QiwuSpmvScalar* host_values;

    const int32_t* device_row_offsets;
    const int32_t* device_column_indices;
    const QiwuSpmvScalar* device_values;
};

struct QiwuSpmvExecutionContext {
    const QiwuSpmvScalar* device_x;
    QiwuSpmvScalar* device_y;
};

struct QiwuSpmvStorage;

inline void qiwu_spmv_check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false);

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false);

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage,
    cudaStream_t stream
) noexcept(false);

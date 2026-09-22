#pragma once

#include <qiwu/gpu_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

enum class QiwuSpmmDataType : uint32_t {
    fp32 = 1,
    fp64 = 2,
};

enum class QiwuSpmmDenseLayout : uint32_t {
    row_major = 1,
    column_major = 2,
};

enum class QiwuSpmmOperation : uint32_t {
    none = 1,
    transpose = 2,
};

#if defined(QIWU_SPMM_FP64)
using QiwuSpmmScalar = double;
constexpr QiwuSpmmDataType QIWU_SPMM_DATA_TYPE = QiwuSpmmDataType::fp64;
#else
using QiwuSpmmScalar = float;
constexpr QiwuSpmmDataType QIWU_SPMM_DATA_TYPE = QiwuSpmmDataType::fp32;
#endif

struct QiwuSpmmProblem {
    QiwuSpmmDataType data_type;
    int64_t rows;
    int64_t cols;
    int64_t rhs_columns;
    int64_t nnz;

    const int32_t* host_row_offsets;
    const int32_t* host_column_indices;
    const QiwuSpmmScalar* host_values;

    const int32_t* device_row_offsets;
    const int32_t* device_column_indices;
    const QiwuSpmmScalar* device_values;
    const QiwuSpmmScalar* device_b;
    QiwuSpmmScalar* device_c;

    int64_t ldb;
    int64_t ldc;
    QiwuSpmmDenseLayout dense_layout;
    QiwuSpmmOperation op_a;
    QiwuSpmmOperation op_b;
};

struct QiwuSpmmExecutionContext {
    QiwuSpmmScalar alpha;
    QiwuSpmmScalar beta;
    bool reset_output = false;
};

struct QiwuSpmmStorage;

using QiwuSpmmStream = cudaStream_t;

inline void qiwu_spmm_check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

inline void qiwu_spmm_check_gpu(cudaError_t status, const char* operation) {
    qiwu_spmm_check_cuda(status, operation);
}

extern "C" QiwuSpmmStorage* qiwu_spmm_preprocess(
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false);

extern "C" void qiwu_spmm_solve(
    QiwuSpmmStorage* storage,
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false);

extern "C" const char* qiwu_spmm_library_version(
    const QiwuSpmmStorage* storage
) noexcept;

extern "C" const char* qiwu_spmm_algorithm(
    const QiwuSpmmStorage* storage
) noexcept;

extern "C" void qiwu_spmm_destroy(
    QiwuSpmmStorage* storage,
    QiwuSpmmStream stream
) noexcept(false);

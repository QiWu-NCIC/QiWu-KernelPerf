#include <qiwu/spmm_plugin.cuh>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <malloc.h>
#endif

#define CUDA_ARCH 0
#if defined(QIWU_BACKEND_HIP)
#define __HIP__ 1
#define __HIPNEED__ 1
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#include "upstream/include/alphasparse.h"
#include "upstream/hip/format/alphasparse_create_csr.h"
#else
#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
#define ALPHA_Complex8 cuFloatComplex
#define ALPHA_Complex16 cuDoubleComplex
#define __CUDA__ 1
#include "upstream/include/alphasparse/type/r_f32_types.h"
#include "upstream/include/alphasparse/type/r_f64_types.h"
#include "upstream/include/alphasparse/type/r_i8_types.h"
#include "upstream/include/alphasparse/type/c_f32_types.h"
#include "upstream/include/alphasparse/type/c_f64_types.h"
#include "upstream/cuda/format/alphasparse_create_csr.h"
#endif

void* alpha_memalign(size_t bytes, size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* pointer = nullptr;
    return posix_memalign(&pointer, alignment, bytes) == 0 ? pointer : nullptr;
#endif
}

struct QiwuAlphaSparseSpmmHandle {
    QiwuSpmmStream stream = nullptr;
    int wavefront_size =
#if defined(QIWU_BACKEND_HIP)
        64;
#else
        32;
#endif
};

#define alphasparseHandle_t QiwuAlphaSparseSpmmHandle*
#if defined(QIWU_BACKEND_HIP)
#include "upstream/hip/kernel/level3/alphasparse_spmm.hip"
#else
#if defined(_WIN32)
#define __attribute__(...)
#endif
#include "upstream/include/alphasparse/common.h"
#define half float
#define half2 cuFloatComplex
#define QIWU_SPMM_REAL_TYPES_ONLY 1
#include "upstream/cuda/kernel/level3/alphasparse_spmm.cu"
#undef QIWU_SPMM_REAL_TYPES_ONLY
#undef half2
#undef half
#endif

#ifndef KERNELPERF_ALPHASPARSE_SPMM_ALGORITHM
#error "select an AlphaSparse SpMM algorithm in the entry source"
#endif

#ifndef KERNELPERF_ALPHASPARSE_SPMM_ALGORITHM_NAME
#error "define the AlphaSparse SpMM algorithm name in the entry source"
#endif

namespace kernelperf_alphasparse_spmm {

constexpr alphasparseDataType value_type =
#if defined(QIWU_SPMM_FP64)
    ALPHA_R_64F;
#else
    ALPHA_R_32F;
#endif

void check(alphasparseStatus_t status, const char* operation) {
    if (status != ALPHA_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with AlphaSparse status " +
            std::to_string(static_cast<int>(status))
        );
    }
}

void validate(const QiwuSpmmProblem* problem, const QiwuSpmmExecutionContext* context) {
    if (!problem || !context || problem->data_type != QIWU_SPMM_DATA_TYPE ||
        problem->rows <= 0 || problem->cols <= 0 || problem->rhs_columns <= 0 ||
        problem->nnz <= 0 || problem->rows > std::numeric_limits<int32_t>::max() ||
        problem->cols > std::numeric_limits<int32_t>::max() ||
        problem->nnz > std::numeric_limits<int32_t>::max() ||
        problem->dense_layout != QiwuSpmmDenseLayout::row_major ||
        problem->op_a != QiwuSpmmOperation::none ||
        problem->op_b != QiwuSpmmOperation::none ||
        problem->ldb != problem->rhs_columns || problem->ldc != problem->rhs_columns) {
        throw std::runtime_error("unsupported AlphaSparse CSR SpMM problem");
    }
}

alphasparseDnMatDescr_t create_dense(
    int64_t rows,
    int64_t columns,
    int64_t leading_dimension,
    void* values
) {
    auto* descriptor = new _alphasparse_dnmat_descr();
    descriptor->init = true;
    descriptor->rows = rows;
    descriptor->cols = columns;
    descriptor->ld = leading_dimension;
    descriptor->values = values;
    descriptor->data_type = value_type;
    descriptor->order = ALPHASPARSE_ORDER_ROW;
    return descriptor;
}

}  // namespace kernelperf_alphasparse_spmm

struct QiwuSpmmStorage {
    QiwuAlphaSparseSpmmHandle handle_value{};
    alphasparseSpMatDescr_t matrix = nullptr;
    alphasparseDnMatDescr_t dense_b = nullptr;
    alphasparseDnMatDescr_t dense_c = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
};

namespace kernelperf_alphasparse_spmm {

void release(QiwuSpmmStorage* storage) noexcept {
    if (!storage) return;
    if (storage->workspace) cudaFree(storage->workspace);
    delete storage->dense_c;
    delete storage->dense_b;
    if (storage->matrix) {
        delete storage->matrix->descr;
        delete storage->matrix;
    }
    delete storage;
}

}  // namespace kernelperf_alphasparse_spmm

extern "C" QiwuSpmmStorage* qiwu_spmm_preprocess(
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false) {
    using namespace kernelperf_alphasparse_spmm;
    validate(problem, context);
    auto* storage = new QiwuSpmmStorage();
    storage->handle_value.stream = stream;
    try {
        check(alphasparseCreateCsr(
            &storage->matrix,
            static_cast<int>(problem->rows),
            static_cast<int>(problem->cols),
            static_cast<int>(problem->nnz),
            const_cast<int32_t*>(problem->device_row_offsets),
            const_cast<int32_t*>(problem->device_column_indices),
            const_cast<QiwuSpmmScalar*>(problem->device_values),
            ALPHA_SPARSE_INDEXTYPE_I32,
            ALPHA_SPARSE_INDEXTYPE_I32,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            value_type
        ), "alphasparseCreateCsr");
        storage->dense_b = create_dense(
            problem->cols,
            problem->rhs_columns,
            problem->ldb,
            const_cast<QiwuSpmmScalar*>(problem->device_b)
        );
        storage->dense_c = create_dense(
            problem->rows,
            problem->rhs_columns,
            problem->ldc,
            problem->device_c
        );
        check(alphasparseSpMM_bufferSize(
            &storage->handle_value,
            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
            &context->alpha,
            storage->matrix,
            storage->dense_b,
            &context->beta,
            storage->dense_c,
            value_type,
            KERNELPERF_ALPHASPARSE_SPMM_ALGORITHM,
            &storage->workspace_size
        ), "alphasparseSpMM_bufferSize");
        if (storage->workspace_size > 0) {
            qiwu_spmm_check_cuda(
                cudaMalloc(&storage->workspace, storage->workspace_size),
                "allocate AlphaSparse SpMM workspace"
            );
        }
#if defined(QIWU_BACKEND_HIP)
        check(alphasparseSpMM_preprocess(
            &storage->handle_value,
            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
            &context->alpha,
            storage->matrix,
            storage->dense_b,
            &context->beta,
            storage->dense_c,
            value_type,
            KERNELPERF_ALPHASPARSE_SPMM_ALGORITHM,
            &storage->workspace_size
        ), "alphasparseSpMM_preprocess");
#endif
        return storage;
    } catch (...) {
        cudaStreamSynchronize(stream);
        release(storage);
        throw;
    }
}

extern "C" void qiwu_spmm_solve(
    QiwuSpmmStorage* storage,
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false) {
    using namespace kernelperf_alphasparse_spmm;
    if (!storage) throw std::runtime_error("invalid AlphaSparse SpMM storage");
    validate(problem, context);
    storage->handle_value.stream = stream;
    if (context->reset_output) {
        qiwu_spmm_check_cuda(
            cudaMemsetAsync(
                problem->device_c,
                0,
                sizeof(QiwuSpmmScalar) * static_cast<size_t>(
                    problem->rows * problem->rhs_columns
                ),
                stream
            ),
            "reset AlphaSparse SpMM output"
        );
    }
    check(alphasparseSpMM(
        &storage->handle_value,
        ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
        ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
        &context->alpha,
        storage->matrix,
        storage->dense_b,
        &context->beta,
        storage->dense_c,
        value_type,
        KERNELPERF_ALPHASPARSE_SPMM_ALGORITHM,
        storage->workspace
    ), "alphasparseSpMM");
}

extern "C" const char* qiwu_spmm_library_version(const QiwuSpmmStorage*) noexcept {
    return "AlphaSparse 248af573c867cf4c7e05d40c611c7c34ebac4715";
}

extern "C" const char* qiwu_spmm_algorithm(const QiwuSpmmStorage*) noexcept {
    return KERNELPERF_ALPHASPARSE_SPMM_ALGORITHM_NAME;
}

extern "C" void qiwu_spmm_destroy(
    QiwuSpmmStorage* storage,
    QiwuSpmmStream stream
) noexcept(false) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    kernelperf_alphasparse_spmm::release(storage);
    qiwu_spmm_check_cuda(status, "synchronize before AlphaSparse SpMM destroy");
}

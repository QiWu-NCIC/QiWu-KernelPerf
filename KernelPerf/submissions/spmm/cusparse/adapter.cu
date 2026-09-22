#include <qiwu/spmm_plugin.cuh>

#include <cusparse.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef KERNELPERF_CUSPARSE_SPMM_ALGORITHM
#error "select a cuSPARSE SpMM algorithm in the entry source"
#endif

#ifndef KERNELPERF_CUSPARSE_SPMM_ALGORITHM_NAME
#error "define the cuSPARSE SpMM algorithm name in the entry source"
#endif

namespace kernelperf_cusparse_spmm {

constexpr cudaDataType value_type =
#if defined(QIWU_SPMM_FP64)
    CUDA_R_64F;
#else
    CUDA_R_32F;
#endif

void check(cusparseStatus_t status, const char* operation) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": " + cusparseGetErrorString(status)
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
        throw std::runtime_error("unsupported cuSPARSE CSR SpMM problem");
    }
}

}  // namespace kernelperf_cusparse_spmm

struct QiwuSpmmStorage {
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnMatDescr_t dense_b = nullptr;
    cusparseDnMatDescr_t dense_c = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    std::string version;
};

namespace kernelperf_cusparse_spmm {

void release(QiwuSpmmStorage* storage) noexcept {
    if (!storage) return;
    if (storage->workspace) cudaFree(storage->workspace);
    if (storage->dense_c) cusparseDestroyDnMat(storage->dense_c);
    if (storage->dense_b) cusparseDestroyDnMat(storage->dense_b);
    if (storage->matrix) cusparseDestroySpMat(storage->matrix);
    if (storage->handle) cusparseDestroy(storage->handle);
    delete storage;
}

}  // namespace kernelperf_cusparse_spmm

extern "C" QiwuSpmmStorage* qiwu_spmm_preprocess(
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false) {
    using namespace kernelperf_cusparse_spmm;
    validate(problem, context);
    auto* storage = new QiwuSpmmStorage();
    try {
        check(cusparseCreate(&storage->handle), "cusparseCreate");
        check(cusparseSetStream(storage->handle, stream), "cusparseSetStream");
        check(cusparseCreateCsr(
            &storage->matrix,
            problem->rows,
            problem->cols,
            problem->nnz,
            const_cast<int32_t*>(problem->device_row_offsets),
            const_cast<int32_t*>(problem->device_column_indices),
            const_cast<QiwuSpmmScalar*>(problem->device_values),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            value_type
        ), "cusparseCreateCsr");
        check(cusparseCreateDnMat(
            &storage->dense_b,
            problem->cols,
            problem->rhs_columns,
            problem->ldb,
            const_cast<QiwuSpmmScalar*>(problem->device_b),
            value_type,
            CUSPARSE_ORDER_ROW
        ), "cusparseCreateDnMat(B)");
        check(cusparseCreateDnMat(
            &storage->dense_c,
            problem->rows,
            problem->rhs_columns,
            problem->ldc,
            problem->device_c,
            value_type,
            CUSPARSE_ORDER_ROW
        ), "cusparseCreateDnMat(C)");
        check(cusparseSpMM_bufferSize(
            storage->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &context->alpha,
            storage->matrix,
            storage->dense_b,
            &context->beta,
            storage->dense_c,
            value_type,
            KERNELPERF_CUSPARSE_SPMM_ALGORITHM,
            &storage->workspace_size
        ), "cusparseSpMM_bufferSize");
        if (storage->workspace_size > 0) {
            qiwu_spmm_check_cuda(
                cudaMalloc(&storage->workspace, storage->workspace_size),
                "allocate cuSPARSE SpMM workspace"
            );
        }
        check(cusparseSpMM_preprocess(
            storage->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &context->alpha,
            storage->matrix,
            storage->dense_b,
            &context->beta,
            storage->dense_c,
            value_type,
            KERNELPERF_CUSPARSE_SPMM_ALGORITHM,
            storage->workspace
        ), "cusparseSpMM_preprocess");
        int major = 0;
        int minor = 0;
        int patch = 0;
        cusparseGetProperty(MAJOR_VERSION, &major);
        cusparseGetProperty(MINOR_VERSION, &minor);
        cusparseGetProperty(PATCH_LEVEL, &patch);
        char version[64];
        std::snprintf(version, sizeof(version), "cuSPARSE %d.%d.%d", major, minor, patch);
        storage->version = version;
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
    using namespace kernelperf_cusparse_spmm;
    if (!storage) throw std::runtime_error("invalid cuSPARSE SpMM storage");
    validate(problem, context);
    check(cusparseSetStream(storage->handle, stream), "cusparseSetStream");
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
            "reset cuSPARSE SpMM output"
        );
    }
    check(cusparseSpMM(
        storage->handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &context->alpha,
        storage->matrix,
        storage->dense_b,
        &context->beta,
        storage->dense_c,
        value_type,
        KERNELPERF_CUSPARSE_SPMM_ALGORITHM,
        storage->workspace
    ), "cusparseSpMM");
}

extern "C" const char* qiwu_spmm_library_version(const QiwuSpmmStorage* storage) noexcept {
    return storage ? storage->version.c_str() : "cuSPARSE unknown";
}

extern "C" const char* qiwu_spmm_algorithm(const QiwuSpmmStorage*) noexcept {
    return KERNELPERF_CUSPARSE_SPMM_ALGORITHM_NAME;
}

extern "C" void qiwu_spmm_destroy(
    QiwuSpmmStorage* storage,
    QiwuSpmmStream stream
) noexcept(false) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    kernelperf_cusparse_spmm::release(storage);
    qiwu_spmm_check_cuda(status, "synchronize before cuSPARSE SpMM destroy");
}

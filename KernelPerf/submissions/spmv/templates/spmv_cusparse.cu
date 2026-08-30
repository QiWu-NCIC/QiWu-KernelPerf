#include <qiwu/spmv_plugin.cuh>

#include <cusparse.h>

#include <stdexcept>
#include <string>

namespace {

void check_cusparse(cusparseStatus_t status, const char* operation) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": " + cusparseGetErrorString(status)
        );
    }
}

constexpr cudaDataType kValueType =
#if defined(QIWU_SPMV_FP64)
    CUDA_R_64F;
#else
    CUDA_R_32F;
#endif

}  // namespace

struct QiwuSpmvStorage {
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t x = nullptr;
    cusparseDnVecDescr_t y = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    QiwuSpmvScalar alpha = static_cast<QiwuSpmvScalar>(1);
    QiwuSpmvScalar beta = static_cast<QiwuSpmvScalar>(0);
};

namespace {

void release_storage(QiwuSpmvStorage* storage) noexcept {
    if (!storage) return;
    if (storage->workspace) cudaFree(storage->workspace);
    if (storage->y) cusparseDestroyDnVec(storage->y);
    if (storage->x) cusparseDestroyDnVec(storage->x);
    if (storage->matrix) cusparseDestroySpMat(storage->matrix);
    if (storage->handle) cusparseDestroy(storage->handle);
    delete storage;
}

}  // namespace

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE) {
        throw std::runtime_error("unsupported Qiwu SpMV plugin input");
    }

    auto* storage = new QiwuSpmvStorage();
    try {
        check_cusparse(cusparseCreate(&storage->handle), "cusparseCreate");
        check_cusparse(cusparseSetStream(storage->handle, stream), "cusparseSetStream");
        check_cusparse(
            cusparseCreateCsr(
                &storage->matrix,
                input->rows,
                input->cols,
                input->nnz,
                const_cast<int32_t*>(input->device_row_offsets),
                const_cast<int32_t*>(input->device_column_indices),
                const_cast<QiwuSpmvScalar*>(input->device_values),
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                kValueType),
            "cusparseCreateCsr");
        check_cusparse(
            cusparseCreateDnVec(
                &storage->x,
                input->cols,
                const_cast<QiwuSpmvScalar*>(context->device_x),
                kValueType),
            "cusparseCreateDnVec x");
        check_cusparse(
            cusparseCreateDnVec(
                &storage->y,
                input->rows,
                context->device_y,
                kValueType),
            "cusparseCreateDnVec y");
        check_cusparse(
            cusparseSpMV_bufferSize(
                storage->handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &storage->alpha,
                storage->matrix,
                storage->x,
                &storage->beta,
                storage->y,
                kValueType,
                CUSPARSE_SPMV_CSR_ALG1,
                &storage->workspace_size),
            "cusparseSpMV_bufferSize");
        if (storage->workspace_size > 0) {
            qiwu_spmv_check_cuda(
                cudaMalloc(&storage->workspace, storage->workspace_size),
                "cudaMalloc cuSPARSE workspace");
        }
#if CUDART_VERSION >= 12040
        check_cusparse(
            cusparseSpMV_preprocess(
                storage->handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &storage->alpha,
                storage->matrix,
                storage->x,
                &storage->beta,
                storage->y,
                kValueType,
                CUSPARSE_SPMV_CSR_ALG1,
                storage->workspace),
            "cusparseSpMV_preprocess");
#endif
        return storage;
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        release_storage(storage);
        throw;
    }
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext*,
    cudaStream_t stream
) noexcept(false) {
    check_cusparse(cusparseSetStream(storage->handle, stream), "cusparseSetStream");
    check_cusparse(
        cusparseSpMV(
            storage->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &storage->alpha,
            storage->matrix,
            storage->x,
            &storage->beta,
            storage->y,
            kValueType,
            CUSPARSE_SPMV_CSR_ALG1,
            storage->workspace),
        "cusparseSpMV");
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage,
    cudaStream_t stream
) noexcept(false) {
    const cudaError_t synchronize_status = cudaStreamSynchronize(stream);
    release_storage(storage);
    qiwu_spmv_check_cuda(synchronize_status, "synchronize before cuSPARSE destroy");
}

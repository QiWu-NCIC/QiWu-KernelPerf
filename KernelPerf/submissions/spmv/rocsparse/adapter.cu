#include <qiwu/spmv_plugin.cuh>

#include <rocsparse/rocsparse.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef KERNELPERF_ROCSPARSE_ALGORITHM
#define KERNELPERF_ROCSPARSE_ALGORITHM rocsparse_spmv_alg_default
#endif

namespace kernelperf_rocsparse {

constexpr rocsparse_datatype value_type =
#if defined(QIWU_SPMV_FP64)
    rocsparse_datatype_f64_r;
#else
    rocsparse_datatype_f32_r;
#endif

void check(rocsparse_status status, const char* operation) {
    if (status != rocsparse_status_success) {
        throw std::runtime_error(
            std::string(operation) + " failed with rocSPARSE status " +
            std::to_string(static_cast<int>(status)));
    }
}

void check_gpu(cudaError_t status, const char* operation) {
    qiwu_spmv_check_cuda(status, operation);
}

}  // namespace kernelperf_rocsparse

struct QiwuSpmvStorage {
    int32_t rows = 0;
    rocsparse_handle handle = nullptr;
    rocsparse_spmat_descr matrix = nullptr;
    rocsparse_dnvec_descr x = nullptr;
    rocsparse_dnvec_descr y = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
};

namespace kernelperf_rocsparse {

void release(QiwuSpmvStorage* storage) noexcept {
    if (!storage) return;
    if (storage->workspace) cudaFree(storage->workspace);
    if (storage->y) rocsparse_destroy_dnvec_descr(storage->y);
    if (storage->x) rocsparse_destroy_dnvec_descr(storage->x);
    if (storage->matrix) rocsparse_destroy_spmat_descr(storage->matrix);
    if (storage->handle) rocsparse_destroy_handle(storage->handle);
    delete storage;
}

void validate(const QiwuSpmvCsrInput* input, const QiwuSpmvExecutionContext* context) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz <= 0 ||
        input->rows > std::numeric_limits<int32_t>::max() ||
        input->cols > std::numeric_limits<int32_t>::max() ||
        input->nnz > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("unsupported rocSPARSE CSR SpMV input");
    }
}

}  // namespace kernelperf_rocsparse

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_rocsparse;
    validate(input, context);
    auto* storage = new QiwuSpmvStorage();
    const int32_t rows = static_cast<int32_t>(input->rows);
    const int32_t cols = static_cast<int32_t>(input->cols);
    const int32_t nnz = static_cast<int32_t>(input->nnz);
    try {
        storage->rows = rows;
        check(rocsparse_create_handle(&storage->handle), "rocsparse_create_handle");
        check(rocsparse_set_stream(storage->handle, stream), "rocsparse_set_stream");
        check(rocsparse_create_csr_descr(
            &storage->matrix, rows, cols, nnz,
            const_cast<int32_t*>(input->device_row_offsets),
            const_cast<int32_t*>(input->device_column_indices),
            const_cast<QiwuSpmvScalar*>(input->device_values),
            rocsparse_indextype_i32, rocsparse_indextype_i32,
            rocsparse_index_base_zero, value_type),
            "rocsparse_create_csr_descr");
        check(rocsparse_create_dnvec_descr(
            &storage->x, cols, const_cast<QiwuSpmvScalar*>(context->device_x), value_type),
            "rocsparse_create_dnvec_descr(x)");
        check(rocsparse_create_dnvec_descr(
            &storage->y, rows, context->device_y, value_type),
            "rocsparse_create_dnvec_descr(y)");

        QiwuSpmvScalar alpha = static_cast<QiwuSpmvScalar>(1);
        QiwuSpmvScalar beta = static_cast<QiwuSpmvScalar>(0);
        check(rocsparse_spmv(
            storage->handle, rocsparse_operation_none, &alpha,
            storage->matrix, storage->x, &beta, storage->y, value_type,
            KERNELPERF_ROCSPARSE_ALGORITHM,
            rocsparse_spmv_stage_buffer_size, &storage->workspace_size, nullptr),
            "rocsparse_spmv(buffer_size)");
        if (storage->workspace_size) {
            check_gpu(cudaMalloc(&storage->workspace, storage->workspace_size),
                      "rocSPARSE workspace allocation");
        }
#if defined(KERNELPERF_ROCSPARSE_NEEDS_PREPROCESS)
        check(rocsparse_spmv(
            storage->handle, rocsparse_operation_none, &alpha,
            storage->matrix, storage->x, &beta, storage->y, value_type,
            KERNELPERF_ROCSPARSE_ALGORITHM,
            rocsparse_spmv_stage_preprocess, &storage->workspace_size, storage->workspace),
            "rocsparse_spmv(preprocess)");
#endif
        check_gpu(cudaStreamSynchronize(stream), "rocSPARSE preprocess synchronize");
        return storage;
    } catch (...) {
        cudaStreamSynchronize(stream);
        release(storage);
        throw;
    }
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_rocsparse;
    if (!storage || !context) throw std::runtime_error("invalid rocSPARSE storage");
    check(rocsparse_set_stream(storage->handle, stream), "rocsparse_set_stream");
    if (context->reset_output) {
        check_gpu(cudaMemsetAsync(
            context->device_y, 0, sizeof(QiwuSpmvScalar) *
            static_cast<size_t>(storage->rows), stream),
            "rocSPARSE output reset");
    }
    QiwuSpmvScalar alpha = static_cast<QiwuSpmvScalar>(1);
    QiwuSpmvScalar beta = static_cast<QiwuSpmvScalar>(0);
    size_t buffer_size = storage->workspace_size;
    check(rocsparse_spmv(
        storage->handle, rocsparse_operation_none, &alpha,
        storage->matrix, storage->x, &beta, storage->y, value_type,
        KERNELPERF_ROCSPARSE_ALGORITHM,
        rocsparse_spmv_stage_compute, &buffer_size, storage->workspace),
        "rocsparse_spmv(compute)");
}

extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage* storage, cudaStream_t stream) noexcept(false) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    kernelperf_rocsparse::release(storage);
    qiwu_spmv_check_cuda(status, "synchronize before rocSPARSE destroy");
}

#include <qiwu/spmv_plugin.cuh>

#include <rocsparse/rocsparse.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace kernelperf_rocsparse_legacy {

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

}  // namespace kernelperf_rocsparse_legacy

struct QiwuSpmvStorage {
    rocsparse_int rows = 0;
    rocsparse_int cols = 0;
    rocsparse_int nnz = 0;
    rocsparse_handle handle = nullptr;
    rocsparse_mat_descr descriptor = nullptr;
    const rocsparse_int* row_offsets = nullptr;
    const rocsparse_int* column_indices = nullptr;
    const QiwuSpmvScalar* values = nullptr;
};

namespace kernelperf_rocsparse_legacy {

void release(QiwuSpmvStorage* storage) noexcept {
    if (!storage) return;
    if (storage->descriptor) rocsparse_destroy_mat_descr(storage->descriptor);
    if (storage->handle) rocsparse_destroy_handle(storage->handle);
    delete storage;
}

void validate(const QiwuSpmvCsrInput* input, const QiwuSpmvExecutionContext* context) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz <= 0 ||
        input->rows > std::numeric_limits<rocsparse_int>::max() ||
        input->cols > std::numeric_limits<rocsparse_int>::max() ||
        input->nnz > std::numeric_limits<rocsparse_int>::max()) {
        throw std::runtime_error("unsupported legacy rocSPARSE CSR SpMV input");
    }
}

}  // namespace kernelperf_rocsparse_legacy

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_rocsparse_legacy;
    validate(input, context);
    auto* storage = new QiwuSpmvStorage();
    try {
        storage->rows = static_cast<rocsparse_int>(input->rows);
        storage->cols = static_cast<rocsparse_int>(input->cols);
        storage->nnz = static_cast<rocsparse_int>(input->nnz);
        storage->row_offsets = input->device_row_offsets;
        storage->column_indices = input->device_column_indices;
        storage->values = input->device_values;
        check(rocsparse_create_handle(&storage->handle), "rocsparse_create_handle");
        check(rocsparse_set_stream(storage->handle, stream), "rocsparse_set_stream");
        check(rocsparse_create_mat_descr(&storage->descriptor), "rocsparse_create_mat_descr");
        check(rocsparse_set_mat_type(storage->descriptor, rocsparse_matrix_type_general),
              "rocsparse_set_mat_type");
        check(rocsparse_set_mat_index_base(storage->descriptor, rocsparse_index_base_zero),
              "rocsparse_set_mat_index_base");
        return storage;
    } catch (...) {
        release(storage);
        throw;
    }
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_rocsparse_legacy;
    if (!storage || !context) {
        throw std::runtime_error("invalid legacy rocSPARSE storage");
    }
    check(rocsparse_set_stream(storage->handle, stream), "rocsparse_set_stream");
    if (context->reset_output) {
        check_gpu(cudaMemsetAsync(
            context->device_y, 0,
            sizeof(QiwuSpmvScalar) * static_cast<size_t>(storage->rows), stream),
            "rocSPARSE output reset");
    }
    const QiwuSpmvScalar alpha = static_cast<QiwuSpmvScalar>(1);
    const QiwuSpmvScalar beta = static_cast<QiwuSpmvScalar>(0);
#if defined(QIWU_SPMV_FP64)
    check(rocsparse_dcsrmv(
#else
    check(rocsparse_scsrmv(
#endif
        storage->handle, rocsparse_operation_none,
        storage->rows, storage->cols, storage->nnz,
        &alpha, storage->descriptor, storage->values,
        storage->row_offsets, storage->column_indices,
        nullptr, context->device_x, &beta, context->device_y),
        "legacy rocsparse_csrmv");
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage, cudaStream_t stream) noexcept(false) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    kernelperf_rocsparse_legacy::release(storage);
    qiwu_spmv_check_cuda(status, "synchronize before legacy rocSPARSE destroy");
}

#include <qiwu/spmv_plugin.cuh>

#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "upstream/CSR5_cuda/anonymouslib_cuda.h"

// Adapter for the public Benchmark_SpMV_using_CSR5 CUDA implementation.
// The vendored CUDA headers only patch zero-block launches on tail-only inputs.
// CSR5 converts columns/values in place, so the adapter gives it private copies.
struct QiwuSpmvStorage {
    using Handle = anonymouslibHandle<int, unsigned int, QiwuSpmvScalar>;

    Handle* handle = nullptr;
    int64_t rows = 0;
    int* row_offsets = nullptr;
    int* column_indices = nullptr;
    QiwuSpmvScalar* values = nullptr;
    bool initialized = false;
};

namespace {

void check_status(int status, const char* operation) {
    if (status != ANONYMOUSLIB_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

void free_device_copy(QiwuSpmvStorage* storage) {
    if (storage->row_offsets) cudaFree(storage->row_offsets);
    if (storage->column_indices) cudaFree(storage->column_indices);
    if (storage->values) cudaFree(storage->values);
    storage->row_offsets = nullptr;
    storage->column_indices = nullptr;
    storage->values = nullptr;
}

int release_handle(QiwuSpmvStorage* storage) noexcept {
    if (!storage || !storage->handle) return ANONYMOUSLIB_SUCCESS;
    int status = ANONYMOUSLIB_SUCCESS;
    if (storage->initialized) {
        status = storage->handle->destroy();
        storage->initialized = false;
    }
    delete storage->handle;
    storage->handle = nullptr;
    return status;
}

void release_storage(QiwuSpmvStorage* storage) noexcept {
    if (!storage) return;
    release_handle(storage);
    free_device_copy(storage);
    delete storage;
}

}  // namespace

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz < 0 ||
        input->rows > std::numeric_limits<int>::max() ||
        input->cols > std::numeric_limits<int>::max() ||
        input->nnz > std::numeric_limits<int>::max()) {
        throw std::runtime_error("unsupported CSR5 input");
    }

    auto storage = std::make_unique<QiwuSpmvStorage>();
    try {
        const size_t row_bytes = static_cast<size_t>(input->rows + 1) * sizeof(int);
        const size_t entry_bytes = static_cast<size_t>(input->nnz) * sizeof(int);
        const size_t value_bytes = static_cast<size_t>(input->nnz) * sizeof(QiwuSpmvScalar);
        qiwu_spmv_check_cuda(cudaMalloc(&storage->row_offsets, row_bytes), "CSR5 row allocation");
        qiwu_spmv_check_cuda(cudaMalloc(&storage->column_indices, entry_bytes), "CSR5 column allocation");
        qiwu_spmv_check_cuda(cudaMalloc(&storage->values, value_bytes), "CSR5 value allocation");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(
            storage->row_offsets, input->device_row_offsets, row_bytes,
            cudaMemcpyDeviceToDevice, stream), "CSR5 row copy");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(
            storage->column_indices, input->device_column_indices, entry_bytes,
            cudaMemcpyDeviceToDevice, stream), "CSR5 column copy");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(
            storage->values, input->device_values, value_bytes,
            cudaMemcpyDeviceToDevice, stream), "CSR5 value copy");
        qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "CSR5 input copy synchronize");

        storage->handle = new QiwuSpmvStorage::Handle(
            static_cast<int>(input->rows), static_cast<int>(input->cols));
        storage->rows = input->rows;
        check_status(storage->handle->inputCSR(
            static_cast<int>(input->nnz), storage->row_offsets,
            storage->column_indices, storage->values), "CSR5 inputCSR");
        storage->initialized = true;
        check_status(storage->handle->setX(const_cast<QiwuSpmvScalar*>(context->device_x)), "CSR5 setX");
        storage->handle->setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
        // The caller performs lifecycle warmup through solve(). The public
        // warmup helper launches a legacy diagnostic kernel and is unreliable
        // on current CUDA toolkits, so it is intentionally not called here.
        check_status(storage->handle->asCSR5(), "CSR5 preprocess");
        return storage.release();
    } catch (...) {
        release_storage(storage.release());
        throw;
    }
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false) {
    if (!storage || !storage->handle || !context) {
        throw std::runtime_error("invalid CSR5 storage");
    }
    qiwu_spmv_check_cuda(
        cudaMemsetAsync(context->device_y, 0,
                        static_cast<size_t>(storage->rows) * sizeof(QiwuSpmvScalar), stream),
        "CSR5 output clear"
    );
    qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "CSR5 output clear synchronize");
    check_status(storage->handle->spmv(QiwuSpmvScalar{1}, context->device_y), "CSR5 SpMV");
    // The public CSR5 API launches on the legacy default stream. Synchronize
    // it before returning so KernelPerf's event on the supplied stream also
    // covers the actual kernel execution.
    qiwu_spmv_check_cuda(cudaStreamSynchronize(nullptr), "CSR5 default stream synchronize");
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage,
    cudaStream_t stream
) noexcept(false) {
    if (!storage) return;
    qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "CSR5 stream synchronize");
    qiwu_spmv_check_cuda(cudaStreamSynchronize(nullptr), "CSR5 default stream synchronize");
    const int status = release_handle(storage);
    free_device_copy(storage);
    delete storage;
    check_status(status, "CSR5 destroy");
}

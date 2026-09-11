#include <qiwu/spmv_plugin.cuh>

#include <cstdint>

// KernelPerf SpMV custom-format template.
//
// The platform supplies the same CSR matrix in host and device memory. This
// baseline consumes the device CSR directly; replace the storage fields and
// preprocess logic when introducing another format.

struct QiwuSpmvStorage {
    int64_t rows;
    const int32_t* device_row_offsets;
    const int32_t* device_column_indices;
    const QiwuSpmvScalar* device_values;
    const QiwuSpmvScalar* device_x;
    QiwuSpmvScalar* device_y;
};

__global__ void spmv_template_csr_kernel(
    int64_t rows,
    const int32_t* row_offsets,
    const int32_t* column_indices,
    const QiwuSpmvScalar* values,
    const QiwuSpmvScalar* x,
    QiwuSpmvScalar* y
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    QiwuSpmvScalar sum = 0;
    for (int32_t index = row_offsets[row]; index < row_offsets[row + 1]; ++index) {
        sum += values[index] * x[column_indices[index]];
    }
    y[row] = sum;
}

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t
) noexcept(false) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE) {
        throw std::runtime_error("unsupported Qiwu SpMV plugin input");
    }
    return new QiwuSpmvStorage{
        input->rows,
        input->device_row_offsets,
        input->device_column_indices,
        input->device_values,
        context->device_x,
        context->device_y,
    };
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext*,
    cudaStream_t stream
) noexcept(false) {
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (storage->rows + block_size - 1) / block_size
    );
    spmv_template_csr_kernel<<<grid_size, block_size, 0, stream>>>(
        storage->rows,
        storage->device_row_offsets,
        storage->device_column_indices,
        storage->device_values,
        storage->device_x,
        storage->device_y
    );
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage,
    cudaStream_t stream
) noexcept(false) {
    const cudaError_t synchronize_status = cudaStreamSynchronize(stream);
    delete storage;
    qiwu_spmv_check_cuda(synchronize_status, "synchronize before template destroy");
}

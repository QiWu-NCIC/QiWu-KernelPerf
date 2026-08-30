#include <qiwu/spmv_plugin.cuh>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kSellSliceHeight = 32;

}  // namespace

struct QiwuSpmvStorage {
    int64_t rows;
    int64_t slice_count;
    int64_t* slice_offsets;
    int32_t* column_indices;
    QiwuSpmvScalar* values;
    const QiwuSpmvScalar* x;
    QiwuSpmvScalar* y;
    std::vector<int64_t> host_slice_offsets;
    std::vector<int32_t> host_column_indices;
    std::vector<QiwuSpmvScalar> host_values;
};

__global__ void spmv_sell_kernel(
    int64_t rows,
    const int64_t* slice_offsets,
    const int32_t* column_indices,
    const QiwuSpmvScalar* values,
    const QiwuSpmvScalar* x,
    QiwuSpmvScalar* y
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t slice = row / kSellSliceHeight;
    const int32_t lane = static_cast<int32_t>(row % kSellSliceHeight);
    const int64_t slice_begin = slice_offsets[slice];
    const int64_t slice_width =
        (slice_offsets[slice + 1] - slice_begin) / kSellSliceHeight;
    QiwuSpmvScalar sum = 0;
    for (int64_t item = 0; item < slice_width; ++item) {
        const int64_t index =
            slice_begin + item * kSellSliceHeight + lane;
        sum += values[index] * x[column_indices[index]];
    }
    y[row] = sum;
}

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream
) noexcept(false) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE) {
        throw std::runtime_error("unsupported Qiwu SpMV plugin input");
    }

    const int32_t* row_offsets = input->host_row_offsets;
    const int32_t* row_columns = input->host_column_indices;
    const QiwuSpmvScalar* row_values = input->host_values;

    const int64_t slice_count =
        (input->rows + kSellSliceHeight - 1) / kSellSliceHeight;
    std::vector<int64_t> slice_offsets(
        static_cast<size_t>(slice_count) + 1,
        0
    );
    for (int64_t slice = 0; slice < slice_count; ++slice) {
        const int64_t row_begin = slice * kSellSliceHeight;
        const int64_t row_end = std::min(
            row_begin + kSellSliceHeight,
            input->rows
        );
        int64_t width = 0;
        for (int64_t row = row_begin; row < row_end; ++row) {
            width = std::max(
                width,
                static_cast<int64_t>(
                    row_offsets[static_cast<size_t>(row) + 1] -
                    row_offsets[static_cast<size_t>(row)]
                )
            );
        }
        slice_offsets[static_cast<size_t>(slice) + 1] =
            slice_offsets[static_cast<size_t>(slice)] +
            width * kSellSliceHeight;
    }

    const size_t sell_nnz = static_cast<size_t>(slice_offsets.back());
    std::vector<int32_t> columns(sell_nnz, 0);
    std::vector<QiwuSpmvScalar> values(sell_nnz, 0);
    for (int64_t row = 0; row < input->rows; ++row) {
        const int64_t slice = row / kSellSliceHeight;
        const int32_t lane = static_cast<int32_t>(row % kSellSliceHeight);
        const int32_t csr_begin = row_offsets[static_cast<size_t>(row)];
        const int32_t csr_end = row_offsets[static_cast<size_t>(row) + 1];
        for (int32_t item = 0; item < csr_end - csr_begin; ++item) {
            const int64_t sell_index =
                slice_offsets[static_cast<size_t>(slice)] +
                static_cast<int64_t>(item) * kSellSliceHeight + lane;
            const int32_t csr_index = csr_begin + item;
            columns[static_cast<size_t>(sell_index)] =
                row_columns[static_cast<size_t>(csr_index)];
            values[static_cast<size_t>(sell_index)] =
                row_values[static_cast<size_t>(csr_index)];
        }
    }

    auto* storage = new QiwuSpmvStorage{
        input->rows,
        slice_count,
        nullptr,
        nullptr,
        nullptr,
        context->device_x,
        context->device_y,
        std::move(slice_offsets),
        std::move(columns),
        std::move(values),
    };
    try {
        qiwu_spmv_check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&storage->slice_offsets),
                storage->host_slice_offsets.size() * sizeof(int64_t)
            ),
            "cudaMalloc SELL slice offsets"
        );
        qiwu_spmv_check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&storage->column_indices),
                storage->host_column_indices.size() * sizeof(int32_t)
            ),
            "cudaMalloc SELL column indices"
        );
        qiwu_spmv_check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&storage->values),
                storage->host_values.size() * sizeof(QiwuSpmvScalar)
            ),
            "cudaMalloc SELL values"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                storage->slice_offsets,
                storage->host_slice_offsets.data(),
                storage->host_slice_offsets.size() * sizeof(int64_t),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy SELL slice offsets"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                storage->column_indices,
                storage->host_column_indices.data(),
                storage->host_column_indices.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy SELL column indices"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                storage->values,
                storage->host_values.data(),
                storage->host_values.size() * sizeof(QiwuSpmvScalar),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy SELL values"
        );
        return storage;
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        cudaFree(storage->slice_offsets);
        cudaFree(storage->column_indices);
        cudaFree(storage->values);
        delete storage;
        throw;
    }
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
    spmv_sell_kernel<<<grid_size, block_size, 0, stream>>>(
        storage->rows,
        storage->slice_offsets,
        storage->column_indices,
        storage->values,
        storage->x,
        storage->y
    );
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage,
    cudaStream_t stream
) noexcept(false) {
    const cudaError_t synchronize_status = cudaStreamSynchronize(stream);
    cudaFree(storage->slice_offsets);
    cudaFree(storage->column_indices);
    cudaFree(storage->values);
    delete storage;
    qiwu_spmv_check_cuda(
        synchronize_status,
        "synchronize before SELL destroy"
    );
}


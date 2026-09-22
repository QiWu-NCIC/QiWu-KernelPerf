#include <qiwu/spmm_plugin.cuh>

#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const int32_t row_offsets[] = {0, 2, 3};
    const int32_t column_indices[] = {0, 2, 1};
    const QiwuSpmmScalar values[] = {2, 1, 3};
    const QiwuSpmmScalar dense_b[] = {4, 5, 6, 7, 8, 9};
    const QiwuSpmmScalar expected[] = {16, 19, 18, 21};
    int32_t* device_rows = nullptr;
    int32_t* device_columns = nullptr;
    QiwuSpmmScalar* device_values = nullptr;
    QiwuSpmmScalar* device_b = nullptr;
    QiwuSpmmScalar* device_c = nullptr;
    cudaStream_t stream = nullptr;
    QiwuSpmmStorage* storage = nullptr;
    int result = 0;
    try {
        qiwu_spmm_check_gpu(cudaStreamCreate(&stream), "create stream");
        qiwu_spmm_check_gpu(cudaMalloc(&device_rows, sizeof(row_offsets)), "allocate rows");
        qiwu_spmm_check_gpu(cudaMalloc(&device_columns, sizeof(column_indices)), "allocate columns");
        qiwu_spmm_check_gpu(cudaMalloc(&device_values, sizeof(values)), "allocate values");
        qiwu_spmm_check_gpu(cudaMalloc(&device_b, sizeof(dense_b)), "allocate B");
        qiwu_spmm_check_gpu(cudaMalloc(&device_c, sizeof(expected)), "allocate C");
        qiwu_spmm_check_gpu(cudaMemcpyAsync(device_rows, row_offsets, sizeof(row_offsets), cudaMemcpyHostToDevice, stream), "copy rows");
        qiwu_spmm_check_gpu(cudaMemcpyAsync(device_columns, column_indices, sizeof(column_indices), cudaMemcpyHostToDevice, stream), "copy columns");
        qiwu_spmm_check_gpu(cudaMemcpyAsync(device_values, values, sizeof(values), cudaMemcpyHostToDevice, stream), "copy values");
        qiwu_spmm_check_gpu(cudaMemcpyAsync(device_b, dense_b, sizeof(dense_b), cudaMemcpyHostToDevice, stream), "copy B");
        qiwu_spmm_check_gpu(cudaMemsetAsync(device_c, 0, sizeof(expected), stream), "clear C");
        const QiwuSpmmProblem problem{
            QIWU_SPMM_DATA_TYPE, 2, 3, 2, 3,
            row_offsets, column_indices, values,
            device_rows, device_columns, device_values, device_b, device_c,
            2, 2, QiwuSpmmDenseLayout::row_major,
            QiwuSpmmOperation::none, QiwuSpmmOperation::none,
        };
        const QiwuSpmmExecutionContext context{1, 0, true};
        storage = qiwu_spmm_preprocess(&problem, &context, stream);
        if (!storage) throw std::runtime_error("preprocess returned null");
        qiwu_spmm_solve(storage, &problem, &context, stream);
        qiwu_spmm_check_gpu(cudaStreamSynchronize(stream), "solve");
        QiwuSpmmScalar output[4]{};
        qiwu_spmm_check_gpu(cudaMemcpy(output, device_c, sizeof(output), cudaMemcpyDeviceToHost), "copy C");
        for (int index = 0; index < 4; ++index) {
            if (!std::isfinite(output[index]) ||
                std::abs(output[index] - expected[index]) > 32 * std::numeric_limits<QiwuSpmmScalar>::epsilon()) {
                throw std::runtime_error("unexpected SpMM output");
            }
        }
        std::cout << "QiWu SpMM plugin: pass, " << qiwu_spmm_library_version(storage) << '\n';
        qiwu_spmm_destroy(storage, stream);
        storage = nullptr;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    if (storage) {
        try { qiwu_spmm_destroy(storage, stream); } catch (...) { result = 1; }
    }
    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_values);
    cudaFree(device_columns);
    cudaFree(device_rows);
    cudaStreamDestroy(stream);
    return result;
}

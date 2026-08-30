#include <qiwu/spmv_plugin.cuh>

#include <cmath>
#include <iostream>

int main() {
    const int32_t row_offsets[] = {0, 2, 3};
    const int32_t column_indices[] = {0, 1, 1};
    const QiwuSpmvScalar values[] = {2, 1, 3};
    const QiwuSpmvScalar x[] = {4, 5};
    const QiwuSpmvScalar expected[] = {13, 15};
    int32_t* device_rows = nullptr;
    int32_t* device_columns = nullptr;
    QiwuSpmvScalar* device_values = nullptr;
    QiwuSpmvScalar* device_x = nullptr;
    QiwuSpmvScalar* device_y = nullptr;
    cudaStream_t stream = nullptr;
    QiwuSpmvStorage* storage = nullptr;
    try {
        qiwu_spmv_check_cuda(cudaStreamCreate(&stream), "create stream");
        qiwu_spmv_check_cuda(cudaMalloc(&device_rows, sizeof(row_offsets)), "allocate rows");
        qiwu_spmv_check_cuda(cudaMalloc(&device_columns, sizeof(column_indices)), "allocate columns");
        qiwu_spmv_check_cuda(cudaMalloc(&device_values, sizeof(values)), "allocate values");
        qiwu_spmv_check_cuda(cudaMalloc(&device_x, sizeof(x)), "allocate x");
        qiwu_spmv_check_cuda(cudaMalloc(&device_y, sizeof(expected)), "allocate y");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(device_rows, row_offsets, sizeof(row_offsets), cudaMemcpyHostToDevice, stream), "copy rows");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(device_columns, column_indices, sizeof(column_indices), cudaMemcpyHostToDevice, stream), "copy columns");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(device_values, values, sizeof(values), cudaMemcpyHostToDevice, stream), "copy values");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(device_x, x, sizeof(x), cudaMemcpyHostToDevice, stream), "copy x");

        const QiwuSpmvCsrInput input{
            QIWU_SPMV_DATA_TYPE, 2, 2, 3,
            row_offsets, column_indices, values,
            device_rows, device_columns, device_values,
        };
        const QiwuSpmvExecutionContext context{device_x, device_y};
        storage = qiwu_spmv_preprocess(&input, &context, stream);
        if (!storage) throw std::runtime_error("preprocess returned null");
        qiwu_spmv_solve(storage, &context, stream);
        qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "solve");
        QiwuSpmvScalar output[2]{};
        qiwu_spmv_check_cuda(cudaMemcpy(output, device_y, sizeof(output), cudaMemcpyDeviceToHost), "copy y");
        qiwu_spmv_destroy(storage, stream);
        storage = nullptr;
        if (std::abs(output[0] - expected[0]) > 1e-5 || std::abs(output[1] - expected[1]) > 1e-5) {
            throw std::runtime_error("unexpected SpMV output");
        }
        std::cout << "qiwu SpMV plugin: pass\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (storage) qiwu_spmv_destroy(storage, stream);
        return 1;
    }
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_values);
    cudaFree(device_columns);
    cudaFree(device_rows);
    cudaStreamDestroy(stream);
    return 0;
}

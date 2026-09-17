#include <qiwu/spmv_plugin.cuh>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    const int32_t rows[] = {0, 2, 3};
    const int32_t columns[] = {0, 1, 1};
    const QiwuSpmvScalar values[] = {2, 1, 3};
    const QiwuSpmvScalar x[] = {4, 5};
    const QiwuSpmvScalar expected[] = {13, 15};
    int32_t* d_rows = nullptr;
    int32_t* d_columns = nullptr;
    QiwuSpmvScalar* d_values = nullptr;
    QiwuSpmvScalar* d_x = nullptr;
    QiwuSpmvScalar* d_y = nullptr;
    QiwuSpmvStream stream = nullptr;
    QiwuSpmvStorage* storage = nullptr;
    try {
        qiwu_spmv_check_cuda(cudaStreamCreate(&stream), "create stream");
        qiwu_spmv_check_cuda(cudaMalloc(&d_rows, sizeof(rows)), "allocate rows");
        qiwu_spmv_check_cuda(cudaMalloc(&d_columns, sizeof(columns)), "allocate columns");
        qiwu_spmv_check_cuda(cudaMalloc(&d_values, sizeof(values)), "allocate values");
        qiwu_spmv_check_cuda(cudaMalloc(&d_x, sizeof(x)), "allocate x");
        qiwu_spmv_check_cuda(cudaMalloc(&d_y, sizeof(expected)), "allocate y");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(d_rows, rows, sizeof(rows), cudaMemcpyHostToDevice, stream), "copy rows");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(d_columns, columns, sizeof(columns), cudaMemcpyHostToDevice, stream), "copy columns");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(d_values, values, sizeof(values), cudaMemcpyHostToDevice, stream), "copy values");
        qiwu_spmv_check_cuda(cudaMemcpyAsync(d_x, x, sizeof(x), cudaMemcpyHostToDevice, stream), "copy x");
        const QiwuSpmvCsrInput input{QIWU_SPMV_DATA_TYPE, 2, 2, 3, rows, columns, values, d_rows, d_columns, d_values};
        const QiwuSpmvExecutionContext context{d_x, d_y};
        storage = qiwu_spmv_preprocess(&input, &context, stream);
        qiwu_spmv_solve(storage, &context, stream);
        qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "solve");
        QiwuSpmvScalar output[2]{};
        qiwu_spmv_check_cuda(cudaMemcpy(output, d_y, sizeof(output), cudaMemcpyDeviceToHost), "copy y");
        qiwu_spmv_destroy(storage, stream);
        storage = nullptr;
        if (std::abs(output[0] - expected[0]) > 1e-5 || std::abs(output[1] - expected[1]) > 1e-5) {
            throw std::runtime_error("unexpected SpMV output");
        }
        std::cout << "qiwu legacy rocSPARSE CSRMV plugin: pass\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (storage) qiwu_spmv_destroy(storage, stream);
        return 1;
    }
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_values);
    cudaFree(d_columns);
    cudaFree(d_rows);
    cudaStreamDestroy(stream);
    return 0;
}

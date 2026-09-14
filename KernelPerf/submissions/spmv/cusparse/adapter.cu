#include <qiwu/spmv_plugin.cuh>

#if defined(QIWU_BACKEND_HIP)
#include <hipsparse/hipsparse.h>

// hipSPARSE keeps the CUDA generic sparse API shape but uses its own public
// names.  Keep the adapter source shared so the benchmark exercises the same
// lifecycle on CUDA and HIP without pretending that a CUDA library is present
// on a non-NVIDIA worker.
#define cudaDataType hipDataType
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F
#define cusparseStatus_t hipsparseStatus_t
#define cusparseHandle_t hipsparseHandle_t
#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#define cusparseDnVecDescr_t hipsparseDnVecDescr_t
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_SPMV_ALG_DEFAULT HIPSPARSE_SPMV_ALG_DEFAULT
#define CUSPARSE_SPMV_COO_ALG1 HIPSPARSE_SPMV_COO_ALG1
#define CUSPARSE_SPMV_COO_ALG2 HIPSPARSE_SPMV_COO_ALG2
#define CUSPARSE_SPMV_CSR_ALG1 HIPSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMV_CSR_ALG2 HIPSPARSE_SPMV_CSR_ALG2
// The DTK 26.04 hipSPARSE API exposes no SELL-specific algorithm enum.  Its
// generic default path is the only honest equivalent for SlicedELL.
#define CUSPARSE_SPMV_SELL_ALG1 HIPSPARSE_SPMV_ALG_DEFAULT
#define cusparseGetErrorString hipsparseGetErrorString
#define cusparseCreate hipsparseCreate
#define cusparseDestroy hipsparseDestroy
#define cusparseSetStream hipsparseSetStream
#define cusparseCreateCoo hipsparseCreateCoo
#define cusparseCreateCsr hipsparseCreateCsr
#define cusparseCreateCsc hipsparseCreateCsc
#define cusparseCreateSlicedEll hipsparseCreateSlicedEll
#define cusparseDestroySpMat hipsparseDestroySpMat
#define cusparseCreateDnVec hipsparseCreateDnVec
#define cusparseDestroyDnVec hipsparseDestroyDnVec
#define cusparseSpMV_bufferSize hipsparseSpMV_bufferSize
#define cusparseSpMV_preprocess hipsparseSpMV_preprocess
#define cusparseSpMV hipsparseSpMV
#else
#include <cusparse.h>
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define KERNELPERF_CUSPARSE_FORMAT_CSR 1
#define KERNELPERF_CUSPARSE_FORMAT_COO 2
#define KERNELPERF_CUSPARSE_FORMAT_CSC 3
#define KERNELPERF_CUSPARSE_FORMAT_SELL 4
#define KERNELPERF_CUSPARSE_FORMAT_SELL_NROWS 5

#ifndef KERNELPERF_CUSPARSE_FORMAT
#error "select KERNELPERF_CUSPARSE_FORMAT in the entry source"
#endif
#ifndef KERNELPERF_CUSPARSE_ALGORITHM
#define KERNELPERF_CUSPARSE_ALGORITHM CUSPARSE_SPMV_ALG_DEFAULT
#endif
#ifndef KERNELPERF_CUSPARSE_SLICE_SIZE
#define KERNELPERF_CUSPARSE_SLICE_SIZE 32
#endif

namespace kernelperf_cusparse {

constexpr cudaDataType value_type =
#if defined(QIWU_SPMV_FP64)
    CUDA_R_64F;
#else
    CUDA_R_32F;
#endif

void check(cusparseStatus_t status, const char* operation) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cusparseGetErrorString(status));
    }
}

struct Allocation {
    void* pointer = nullptr;
    void reset() noexcept {
        if (pointer) cudaFree(pointer);
        pointer = nullptr;
    }
};

}  // namespace kernelperf_cusparse

namespace kernelperf_cusparse {

void upload(Allocation& allocation, const void* data, size_t bytes, cudaStream_t stream, const char* name) {
    if (!bytes) throw std::runtime_error(std::string(name) + ": empty buffer");
    qiwu_spmv_check_cuda(cudaMalloc(&allocation.pointer, bytes), name);
    try {
        qiwu_spmv_check_cuda(cudaMemcpyAsync(
            allocation.pointer, data, bytes, cudaMemcpyHostToDevice, stream), name);
    } catch (...) {
        allocation.reset();
        throw;
    }
}

}  // namespace kernelperf_cusparse

struct QiwuSpmvStorage {
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t x = nullptr;
    cusparseDnVecDescr_t y = nullptr;
    kernelperf_cusparse::Allocation primary;
    kernelperf_cusparse::Allocation secondary;
    kernelperf_cusparse::Allocation values;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    QiwuSpmvScalar alpha = static_cast<QiwuSpmvScalar>(1);
    QiwuSpmvScalar beta = static_cast<QiwuSpmvScalar>(0);
};

namespace kernelperf_cusparse {

void release(QiwuSpmvStorage* storage) noexcept {
    if (!storage) return;
    if (storage->workspace) cudaFree(storage->workspace);
    storage->primary.reset();
    storage->secondary.reset();
    storage->values.reset();
    if (storage->y) cusparseDestroyDnVec(storage->y);
    if (storage->x) cusparseDestroyDnVec(storage->x);
    if (storage->matrix) cusparseDestroySpMat(storage->matrix);
    if (storage->handle) cusparseDestroy(storage->handle);
    delete storage;
}

void validate(const QiwuSpmvCsrInput* input, const QiwuSpmvExecutionContext* context) {
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz <= 0 ||
        input->rows > std::numeric_limits<int32_t>::max() ||
        input->cols > std::numeric_limits<int32_t>::max() ||
        input->nnz > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("unsupported cuSPARSE SpMV input");
    }
}

void build_coo(const QiwuSpmvCsrInput* input, cudaStream_t stream, QiwuSpmvStorage& storage) {
    std::vector<int32_t> rows(static_cast<size_t>(input->nnz));
    std::vector<int32_t> columns(static_cast<size_t>(input->nnz));
    std::vector<QiwuSpmvScalar> values(static_cast<size_t>(input->nnz));
    for (int64_t row = 0; row < input->rows; ++row) {
        for (int32_t index = input->host_row_offsets[row]; index < input->host_row_offsets[row + 1]; ++index) {
            rows[index] = static_cast<int32_t>(row);
            columns[index] = input->host_column_indices[index];
            values[index] = input->host_values[index];
        }
    }
    upload(storage.primary, rows.data(), rows.size() * sizeof(int32_t), stream, "COO row buffer");
    upload(storage.secondary, columns.data(), columns.size() * sizeof(int32_t), stream, "COO column buffer");
    upload(storage.values, values.data(), values.size() * sizeof(QiwuSpmvScalar), stream, "COO value buffer");
    check(cusparseCreateCoo(&storage.matrix, input->rows, input->cols, input->nnz,
        storage.primary.pointer, storage.secondary.pointer, storage.values.pointer,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, value_type), "cusparseCreateCoo");
}

void build_csc(const QiwuSpmvCsrInput* input, cudaStream_t stream, QiwuSpmvStorage& storage) {
    std::vector<int32_t> offsets(static_cast<size_t>(input->cols) + 1, 0);
    std::vector<int32_t> rows(static_cast<size_t>(input->nnz));
    std::vector<QiwuSpmvScalar> values(static_cast<size_t>(input->nnz));
    for (int64_t index = 0; index < input->nnz; ++index) ++offsets[input->host_column_indices[index] + 1];
    for (int64_t col = 0; col < input->cols; ++col) offsets[col + 1] += offsets[col];
    std::vector<int32_t> cursor = offsets;
    for (int64_t row = 0; row < input->rows; ++row) {
        for (int32_t index = input->host_row_offsets[row]; index < input->host_row_offsets[row + 1]; ++index) {
            const int32_t col = input->host_column_indices[index];
            const int32_t destination = cursor[col]++;
            rows[destination] = static_cast<int32_t>(row);
            values[destination] = input->host_values[index];
        }
    }
    upload(storage.primary, offsets.data(), offsets.size() * sizeof(int32_t), stream, "CSC offset buffer");
    upload(storage.secondary, rows.data(), rows.size() * sizeof(int32_t), stream, "CSC row buffer");
    upload(storage.values, values.data(), values.size() * sizeof(QiwuSpmvScalar), stream, "CSC value buffer");
    check(cusparseCreateCsc(&storage.matrix, input->rows, input->cols, input->nnz,
        storage.primary.pointer, storage.secondary.pointer, storage.values.pointer,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, value_type), "cusparseCreateCsc");
}

void build_sell_nrows(const QiwuSpmvCsrInput* input, cudaStream_t stream, QiwuSpmvStorage& storage) {
#if KERNELPERF_CUSPARSE_FORMAT == KERNELPERF_CUSPARSE_FORMAT_SELL_NROWS
    const int64_t slice_size = input->rows;
#else
    constexpr int64_t slice_size = KERNELPERF_CUSPARSE_SLICE_SIZE;
#endif
    const int64_t slices = (input->rows + slice_size - 1) / slice_size;
    std::vector<int32_t> offsets(static_cast<size_t>(slices) + 1, 0);
    for (int64_t slice = 0; slice < slices; ++slice) {
        const int64_t begin = slice * slice_size;
        const int64_t end = std::min(input->rows, begin + slice_size);
        int64_t width = 0;
        for (int64_t row = begin; row < end; ++row)
            width = std::max<int64_t>(width, input->host_row_offsets[row + 1] - input->host_row_offsets[row]);
        const int64_t next = static_cast<int64_t>(offsets[slice]) + width * slice_size;
        if (next > std::numeric_limits<int32_t>::max()) throw std::runtime_error("SELL offsets exceed int32");
        offsets[slice + 1] = static_cast<int32_t>(next);
    }
    const size_t padded_nnz = static_cast<size_t>(offsets.back());
    std::vector<int32_t> columns(padded_nnz, -1);
    std::vector<QiwuSpmvScalar> values(padded_nnz, static_cast<QiwuSpmvScalar>(0));
    for (int64_t row = 0; row < input->rows; ++row) {
        const int64_t slice = row / slice_size;
        const int64_t lane = row % slice_size;
        const int32_t begin = input->host_row_offsets[row];
        const int32_t end = input->host_row_offsets[row + 1];
        for (int32_t item = 0; item < end - begin; ++item) {
            const size_t destination = static_cast<size_t>(offsets[slice]) + item * slice_size + lane;
            columns[destination] = input->host_column_indices[begin + item];
            values[destination] = input->host_values[begin + item];
        }
    }
    upload(storage.primary, offsets.data(), offsets.size() * sizeof(int32_t), stream, "SELL offsets buffer");
    upload(storage.secondary, columns.data(), columns.size() * sizeof(int32_t), stream, "SELL column buffer");
    upload(storage.values, values.data(), values.size() * sizeof(QiwuSpmvScalar), stream, "SELL value buffer");
    check(cusparseCreateSlicedEll(&storage.matrix, input->rows, input->cols, input->nnz,
        static_cast<int64_t>(padded_nnz), slice_size, storage.primary.pointer,
        storage.secondary.pointer, storage.values.pointer, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, value_type), "cusparseCreateSlicedEll");
}

}  // namespace kernelperf_cusparse

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input, const QiwuSpmvExecutionContext* context, cudaStream_t stream
) noexcept(false) {
    using namespace kernelperf_cusparse;
    validate(input, context);
    auto* storage = new QiwuSpmvStorage();
    try {
        check(cusparseCreate(&storage->handle), "cusparseCreate");
        check(cusparseSetStream(storage->handle, stream), "cusparseSetStream");
#if KERNELPERF_CUSPARSE_FORMAT == KERNELPERF_CUSPARSE_FORMAT_CSR
        check(cusparseCreateCsr(&storage->matrix, input->rows, input->cols, input->nnz,
            const_cast<int32_t*>(input->device_row_offsets), const_cast<int32_t*>(input->device_column_indices),
            const_cast<QiwuSpmvScalar*>(input->device_values), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, value_type), "cusparseCreateCsr");
#elif KERNELPERF_CUSPARSE_FORMAT == KERNELPERF_CUSPARSE_FORMAT_COO
        build_coo(input, stream, *storage);
#elif KERNELPERF_CUSPARSE_FORMAT == KERNELPERF_CUSPARSE_FORMAT_CSC
        build_csc(input, stream, *storage);
#elif KERNELPERF_CUSPARSE_FORMAT == KERNELPERF_CUSPARSE_FORMAT_SELL || KERNELPERF_CUSPARSE_FORMAT == KERNELPERF_CUSPARSE_FORMAT_SELL_NROWS
        build_sell_nrows(input, stream, *storage);
#else
#error "unsupported KERNELPERF_CUSPARSE_FORMAT"
#endif
        check(cusparseCreateDnVec(&storage->x, input->cols, const_cast<QiwuSpmvScalar*>(context->device_x), value_type), "cusparseCreateDnVec x");
        check(cusparseCreateDnVec(&storage->y, input->rows, context->device_y, value_type), "cusparseCreateDnVec y");
        QiwuSpmvScalar alpha = static_cast<QiwuSpmvScalar>(1), beta = static_cast<QiwuSpmvScalar>(0);
        size_t bytes = 0;
        check(cusparseSpMV_bufferSize(storage->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, storage->matrix, storage->x, &beta, storage->y, value_type,
            KERNELPERF_CUSPARSE_ALGORITHM, &bytes), "cusparseSpMV_bufferSize");
        storage->alpha = alpha;
        storage->beta = beta;
        storage->workspace_size = bytes;
        if (bytes) qiwu_spmv_check_cuda(cudaMalloc(&storage->workspace, bytes), "cuSPARSE workspace allocation");
#if defined(QIWU_BACKEND_HIP) || CUDART_VERSION >= 12040
        check(cusparseSpMV_preprocess(storage->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &storage->alpha, storage->matrix, storage->x, &storage->beta, storage->y,
            value_type, KERNELPERF_CUSPARSE_ALGORITHM, storage->workspace), "cusparseSpMV_preprocess");
#endif
        qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "cuSPARSE preprocess synchronize");
        return storage;
    } catch (...) {
        cudaStreamSynchronize(stream);
        release(storage);
        throw;
    }
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage, const QiwuSpmvExecutionContext*, cudaStream_t stream
) noexcept(false) {
    using namespace kernelperf_cusparse;
    if (!storage) throw std::runtime_error("invalid cuSPARSE storage");
    check(cusparseSetStream(storage->handle, stream), "cusparseSetStream");
    check(cusparseSpMV(storage->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &storage->alpha, storage->matrix, storage->x, &storage->beta, storage->y,
        value_type, KERNELPERF_CUSPARSE_ALGORITHM, storage->workspace), "cusparseSpMV");
}

extern "C" void qiwu_spmv_destroy(QiwuSpmvStorage* storage, cudaStream_t stream) noexcept(false) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    kernelperf_cusparse::release(storage);
    qiwu_spmv_check_cuda(status, "synchronize before cuSPARSE destroy");
}

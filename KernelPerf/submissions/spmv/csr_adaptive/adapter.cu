#include <qiwu/spmv_plugin.cuh>

#include <qiwu/gpu_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// CUDA port of clSPARSE's CSR-Adaptive policy. The fixed constants mirror the
// upstream OpenCL worker configuration; only the execution backend is changed.
namespace kernelperf_csr_adaptive {

constexpr int kWorkGroupSize = 256;
constexpr int kBlockSize = 1024;
constexpr int kRowsForVector = 1;
constexpr int kBlockMultiplier = 3;
constexpr int kRowBits = 32;
constexpr int kWorkGroupBits = 24;
constexpr int kVectorThreshold = kBlockMultiplier * kRowsForVector * kRowBits;

static_assert(kWorkGroupSize == 256 && kRowBits == 32 && kWorkGroupBits == 24);
static_assert(kBlockSize == 1024 && kVectorThreshold < kBlockSize);

struct RowPlan {
    int32_t begin;
    int32_t end;
    uint8_t vector_mode;
};

struct RowBlock {
    int32_t row_begin;
    int32_t row_end;
    uint8_t vector_mode;
};

}  // namespace kernelperf_csr_adaptive

struct QiwuSpmvStorage {
    int64_t rows = 0;
    const int32_t* row_offsets = nullptr;
    const int32_t* column_indices = nullptr;
    const QiwuSpmvScalar* values = nullptr;
    const QiwuSpmvScalar* x = nullptr;
    QiwuSpmvScalar* y = nullptr;
    kernelperf_csr_adaptive::RowPlan* plan = nullptr;
    kernelperf_csr_adaptive::RowBlock* blocks = nullptr;
    int32_t block_count = 0;
};

namespace kernelperf_csr_adaptive {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void adaptive_kernel(
    int32_t block_count, const RowBlock* blocks, const RowPlan* plan,
    const int32_t* columns, const QiwuSpmvScalar* values, const QiwuSpmvScalar* x,
    QiwuSpmvScalar* y) {
    if (blockIdx.x >= block_count) return;
    const RowBlock block = blocks[blockIdx.x];
    const int lane = threadIdx.x & 31;
    if (!block.vector_mode) {
        const int64_t row = static_cast<int64_t>(block.row_begin) + threadIdx.x;
        if (row >= block.row_end) return;
        const RowPlan row_plan = plan[row];
        QiwuSpmvScalar sum = 0;
        for (int32_t index = row_plan.begin; index < row_plan.end; ++index) {
            sum += values[index] * x[columns[index]];
        }
        y[row] = sum;
        return;
    }
    const int64_t row = static_cast<int64_t>(block.row_begin) + threadIdx.x / 32;
    if (row >= block.row_end) return;
    const RowPlan row_plan = plan[row];
    QiwuSpmvScalar sum = 0;
    for (int32_t index = row_plan.begin + lane; index < row_plan.end; index += 32) {
        sum += values[index] * x[columns[index]];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

}  // namespace kernelperf_csr_adaptive

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_csr_adaptive;
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz < 0 ||
        input->rows > std::numeric_limits<int32_t>::max() ||
        input->nnz > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("invalid CSR-Adaptive CUDA input");
    }
    auto* storage = new QiwuSpmvStorage();
    storage->rows = input->rows;
    storage->row_offsets = input->device_row_offsets;
    storage->column_indices = input->device_column_indices;
    storage->values = input->device_values;
    storage->x = context->device_x;
    storage->y = context->device_y;
    try {
        std::vector<RowPlan> host_plan(static_cast<size_t>(input->rows));
        std::vector<RowBlock> host_blocks;
        host_blocks.reserve(static_cast<size_t>(input->rows));
        for (int64_t row = 0; row < input->rows; ++row) {
            const int32_t begin = input->host_row_offsets[row];
            const int32_t end = input->host_row_offsets[row + 1];
            if (begin < 0 || end < begin || end > input->nnz) {
                throw std::runtime_error("CSR-Adaptive CSR row offsets are invalid");
            }
            const int32_t length = end - begin;
            host_plan[static_cast<size_t>(row)] = {
                begin, end,
                static_cast<uint8_t>(kRowsForVector == 1 &&
                                     length > kVectorThreshold)};
        }
        int32_t row = 0;
        while (row < input->rows) {
            if (host_plan[static_cast<size_t>(row)].vector_mode) {
                host_blocks.push_back({row, row + 1, 1});
                ++row;
                continue;
            }
            const int32_t block_begin = row;
            int32_t entries = 0;
            while (row < input->rows &&
                   !host_plan[static_cast<size_t>(row)].vector_mode) {
                const RowPlan& item = host_plan[static_cast<size_t>(row)];
                const int32_t length = item.end - item.begin;
                if (row - block_begin >= kWorkGroupSize) break;
                if (row != block_begin && entries + length > kBlockSize) break;
                entries += length;
                ++row;
            }
            host_blocks.push_back({block_begin, row, 0});
        }
        if (host_blocks.empty()) throw std::runtime_error("CSR-Adaptive produced no row blocks");
        check(cudaMalloc(reinterpret_cast<void**>(&storage->plan),
                         host_plan.size() * sizeof(RowPlan)), "CSR-Adaptive plan allocation");
        check(cudaMalloc(reinterpret_cast<void**>(&storage->blocks),
                         host_blocks.size() * sizeof(RowBlock)), "CSR-Adaptive row-block allocation");
        check(cudaMemcpyAsync(storage->plan, host_plan.data(),
                              host_plan.size() * sizeof(RowPlan),
                              cudaMemcpyHostToDevice, stream), "CSR-Adaptive plan upload");
        check(cudaMemcpyAsync(storage->blocks, host_blocks.data(),
                              host_blocks.size() * sizeof(RowBlock),
                              cudaMemcpyHostToDevice, stream), "CSR-Adaptive row-block upload");
        storage->block_count = static_cast<int32_t>(host_blocks.size());
        check(cudaMemsetAsync(storage->y, 0,
                              static_cast<size_t>(storage->rows) * sizeof(QiwuSpmvScalar), stream),
              "CSR-Adaptive output clear");
        return storage;
    } catch (...) {
        if (storage->blocks) cudaFree(storage->blocks);
        if (storage->plan) cudaFree(storage->plan);
        delete storage;
        throw;
    }
}

extern "C" void qiwu_spmv_solve(
    QiwuSpmvStorage* storage,
    const QiwuSpmvExecutionContext*,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_csr_adaptive;
    if (!storage) throw std::runtime_error("CSR-Adaptive storage is null");
    adaptive_kernel<<<storage->block_count, kWorkGroupSize, 0, stream>>>(
        storage->block_count, storage->blocks, storage->plan,
        storage->column_indices, storage->values, storage->x, storage->y);
    check(cudaGetLastError(), "launch CSR-Adaptive CUDA kernel");
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage, cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_csr_adaptive;
    if (!storage) return;
    check(cudaStreamSynchronize(stream), "CSR-Adaptive stream synchronize");
    if (storage->blocks) check(cudaFree(storage->blocks), "CSR-Adaptive row-block release");
    if (storage->plan) check(cudaFree(storage->plan), "CSR-Adaptive plan release");
    delete storage;
}

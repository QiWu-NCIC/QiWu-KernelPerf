#include <qiwu/spmv_plugin.cuh>

#include <cuda_runtime.h>
#include <ghost.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef KERNELPERF_GHOST_SIGMA
#define KERNELPERF_GHOST_SIGMA 128
#endif

struct QiwuSpmvStorage {
    ghost_context* context = nullptr;
    ghost_sparsemat* matrix = nullptr;
    ghost_densemat* x = nullptr;
    ghost_densemat* y = nullptr;
    std::vector<QiwuSpmvScalar> host_x;
    std::vector<QiwuSpmvScalar> host_values;
    std::vector<ghost_gidx> ghost_columns;
    std::vector<ghost_lidx> ghost_row_offsets;
    std::vector<int32_t> original_rows;
    int32_t* device_original_rows = nullptr;
};

namespace kernelperf_ghost {

static_assert(KERNELPERF_GHOST_SIGMA > 0, "GHOST sigma must be positive");

void check(ghost_error status, const char* operation) {
    if (status != GHOST_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with GHOST status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

ghost_datatype datatype() {
#if defined(QIWU_SPMV_FP64)
    return static_cast<ghost_datatype>(GHOST_DT_DOUBLE | GHOST_DT_REAL);
#else
    return static_cast<ghost_datatype>(GHOST_DT_FLOAT | GHOST_DT_REAL);
#endif
}

int csr_row(
    ghost_gidx row,
    ghost_lidx* row_length,
    ghost_gidx* columns,
    void* values,
    void* argument) {
    auto* storage = static_cast<QiwuSpmvStorage*>(argument);
    if (!storage || !row_length || row < 0 ||
        static_cast<size_t>(row + 1) >= storage->ghost_row_offsets.size()) {
        return 1;
    }
    const int32_t original_row = storage->original_rows[static_cast<size_t>(row)];
    const ghost_lidx begin = storage->ghost_row_offsets[static_cast<size_t>(original_row)];
    const ghost_lidx end = storage->ghost_row_offsets[static_cast<size_t>(original_row) + 1];
    *row_length = end - begin;
    if (!columns || !values) return 0;
    auto* typed_values = static_cast<QiwuSpmvScalar*>(values);
    for (ghost_lidx index = begin; index < end; ++index) {
        const size_t source = static_cast<size_t>(index);
        const size_t target = static_cast<size_t>(index - begin);
        columns[target] = storage->ghost_columns[source];
        typed_values[target] = storage->host_values[source];
    }
    return 0;
}

__global__ void restore_row_order(
    const QiwuSpmvScalar* sorted,
    QiwuSpmvScalar* original,
    const int32_t* original_rows,
    int64_t rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) original[original_rows[row]] = sorted[row];
}

void release(QiwuSpmvStorage* storage) noexcept {
    if (!storage) return;
    // GHOST 22a004db registers shared map/task ownership that is released by
    // its process teardown.  Explicit object destruction double-frees that
    // state on CUDA 12.x; the benchmark template uses quick_exit after the
    // flushed measurement, so leaking the context is intentional and scoped
    // to this one-matrix process.
    if (storage->device_original_rows) cudaFree(storage->device_original_rows);
    delete storage;
}

}  // namespace kernelperf_ghost

extern "C" QiwuSpmvStorage* qiwu_spmv_preprocess(
    const QiwuSpmvCsrInput* input,
    const QiwuSpmvExecutionContext* context,
    cudaStream_t stream) noexcept(false) {
    using namespace kernelperf_ghost;
    if (!input || !context || input->data_type != QIWU_SPMV_DATA_TYPE ||
        input->rows <= 0 || input->cols <= 0 || input->nnz <= 0 ||
        input->rows > std::numeric_limits<int32_t>::max() ||
        input->rows > GHOST_GIDX_MAX || input->cols > GHOST_GIDX_MAX ||
        input->nnz > GHOST_GIDX_MAX) {
        throw std::runtime_error("invalid GHOST SpMV input");
    }
    auto* storage = new QiwuSpmvStorage();
    try {
        check(ghost_type_set(GHOST_TYPE_CUDA), "ghost_type_set");
        if (!ghost_initialized()) {
            char program[] = "kernelperf-ghost";
            char* argv[] = {program, nullptr};
            check(ghost_init(1, argv), "ghost_init");
        }
        check(ghost_context_create(&storage->context, static_cast<ghost_gidx>(input->rows),
                                   static_cast<ghost_gidx>(input->cols), GHOST_CONTEXT_DEFAULT,
                                   MPI_COMM_SELF, 1.0), "ghost_context_create");

        ghost_sparsemat_traits traits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
        traits.datatype = datatype();
        traits.C = 32;
        traits.sortScope = 1;
        traits.flags = GHOST_SPARSEMAT_DEVICE;
        check(ghost_sparsemat_create(&storage->matrix, storage->context, &traits, 1),
              "ghost_sparsemat_create");
        storage->ghost_columns.resize(static_cast<size_t>(input->nnz));
        for (int64_t index = 0; index < input->nnz; ++index) {
            storage->ghost_columns[static_cast<size_t>(index)] =
                static_cast<ghost_gidx>(input->host_column_indices[index]);
        }
        storage->ghost_row_offsets.resize(static_cast<size_t>(input->rows + 1));
        storage->host_values.resize(static_cast<size_t>(input->nnz));
        for (int64_t index = 0; index < input->nnz; ++index) {
            storage->host_values[static_cast<size_t>(index)] = input->host_values[index];
        }
        for (int64_t row = 0; row <= input->rows; ++row) {
            storage->ghost_row_offsets[static_cast<size_t>(row)] =
                static_cast<ghost_lidx>(input->host_row_offsets[row]);
        }
        storage->original_rows.resize(static_cast<size_t>(input->rows));
        std::iota(storage->original_rows.begin(), storage->original_rows.end(), int32_t{0});
        const size_t sort_scope = static_cast<size_t>(KERNELPERF_GHOST_SIGMA);
        for (size_t begin = 0; begin < storage->original_rows.size(); begin += sort_scope) {
            const size_t end = std::min(begin + sort_scope, storage->original_rows.size());
            std::stable_sort(storage->original_rows.begin() + begin,
                             storage->original_rows.begin() + end,
                             [&](int32_t lhs, int32_t rhs) {
                const ghost_lidx lhs_length =
                    storage->ghost_row_offsets[static_cast<size_t>(lhs) + 1] -
                    storage->ghost_row_offsets[static_cast<size_t>(lhs)];
                const ghost_lidx rhs_length =
                    storage->ghost_row_offsets[static_cast<size_t>(rhs) + 1] -
                    storage->ghost_row_offsets[static_cast<size_t>(rhs)];
                return lhs_length > rhs_length;
            });
        }
        const size_t permutation_bytes = storage->original_rows.size() * sizeof(int32_t);
        check_cuda(cudaMalloc(&storage->device_original_rows, permutation_bytes),
                   "allocate GHOST row permutation");
        check_cuda(cudaMemcpyAsync(storage->device_original_rows, storage->original_rows.data(),
                                   permutation_bytes, cudaMemcpyHostToDevice, stream),
                   "copy GHOST row permutation");
        ghost_lidx max_row_length = 0;
        for (int64_t row = 0; row < input->rows; ++row) {
            const ghost_lidx row_length =
                storage->ghost_row_offsets[static_cast<size_t>(row) + 1] -
                storage->ghost_row_offsets[static_cast<size_t>(row)];
            if (row_length > max_row_length) max_row_length = row_length;
        }
        ghost_sparsemat_src_rowfunc source = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
        source.func = csr_row;
        source.maxrowlen = max_row_length;
        source.gnrows = static_cast<ghost_gidx>(input->rows);
        source.gncols = static_cast<ghost_gidx>(input->cols);
        source.arg = storage;
        check(ghost_sparsemat_init_rowfunc(storage->matrix, &source, MPI_COMM_SELF, 1.0),
              "ghost_sparsemat_init_rowfunc");

        ghost_densemat_traits dense_traits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        dense_traits.ncols = 1;
        dense_traits.storage = GHOST_DENSEMAT_COLMAJOR;
        dense_traits.datatype = datatype();
        dense_traits.location = static_cast<ghost_location>(GHOST_LOCATION_HOST | GHOST_LOCATION_DEVICE);
        dense_traits.compute_at = GHOST_LOCATION_DEVICE;
        dense_traits.compute_with = GHOST_IMPLEMENTATION_CUDA;
        check(ghost_densemat_create(&storage->x,
                                    ghost_context_map(storage->context, GHOST_MAP_COL), dense_traits),
              "ghost_densemat_create x");
        check(ghost_densemat_create(&storage->y,
                                    ghost_context_map(storage->context, GHOST_MAP_ROW), dense_traits),
              "ghost_densemat_create y");
        int need_initialization = 0;
        check(ghost_densemat_malloc(storage->x, &need_initialization),
              "ghost_densemat_malloc x");
        check(ghost_densemat_malloc(storage->y, &need_initialization),
              "ghost_densemat_malloc y");
        storage->host_x.resize(static_cast<size_t>(input->cols));
        check_cuda(cudaMemcpyAsync(storage->host_x.data(), context->device_x,
                                   storage->host_x.size() * sizeof(QiwuSpmvScalar),
                                   cudaMemcpyDeviceToHost, stream), "copy GHOST x");
        check_cuda(cudaStreamSynchronize(stream), "synchronize GHOST x");
        if (!storage->x->val || !storage->y->val) {
            throw std::runtime_error("GHOST did not allocate host densemat storage");
        }
        std::memset(storage->x->val, 0,
                    static_cast<size_t>(DM_NROWSPAD(storage->x)) * sizeof(QiwuSpmvScalar));
        std::memcpy(storage->x->val, storage->host_x.data(),
                    storage->host_x.size() * sizeof(QiwuSpmvScalar));
        std::memset(storage->y->val, 0,
                    static_cast<size_t>(DM_NROWSPAD(storage->y)) * sizeof(QiwuSpmvScalar));
        check(ghost_densemat_upload(storage->x), "ghost_densemat_upload x");
        check(ghost_densemat_upload(storage->y), "ghost_densemat_upload y");
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
    using namespace kernelperf_ghost;
    if (!storage || !context) throw std::runtime_error("GHOST storage is null");
    ghost_spmv_opts options = GHOST_SPMV_OPTS_INITIALIZER;
    options.blocksz = 1;
    check(ghost_cu_sell_spmv_selector(storage->y, storage->matrix, storage->x, options),
          "ghost_cu_sell_spmv_selector");
    check_cuda(cudaDeviceSynchronize(), "synchronize GHOST CUDA kernel");
    constexpr int threads = 256;
    const int64_t rows = static_cast<int64_t>(DM_NROWS(storage->y));
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    restore_row_order<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const QiwuSpmvScalar*>(storage->y->cu_val), context->device_y,
        storage->device_original_rows, rows);
    check_cuda(cudaGetLastError(), "restore GHOST row order");
    check_cuda(cudaStreamSynchronize(stream), "synchronize GHOST row order");
}

extern "C" void qiwu_spmv_destroy(
    QiwuSpmvStorage* storage, cudaStream_t stream) noexcept(false) {
    if (storage) kernelperf_ghost::check_cuda(
        cudaStreamSynchronize(stream), "synchronize GHOST output copy");
    kernelperf_ghost::release(storage);
}

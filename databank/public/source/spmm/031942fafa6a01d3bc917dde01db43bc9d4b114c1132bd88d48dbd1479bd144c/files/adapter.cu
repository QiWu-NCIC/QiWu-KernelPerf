#include <qiwu/spmm_plugin.cuh>

#include <rocsparse/rocsparse.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef KERNELPERF_ROCSPARSE_SPMM_ALGORITHM
#error "select a rocSPARSE SpMM algorithm in the entry source"
#endif

#ifndef KERNELPERF_ROCSPARSE_SPMM_ALGORITHM_NAME
#error "define the rocSPARSE SpMM algorithm name in the entry source"
#endif

namespace kernelperf_rocsparse_spmm {

constexpr rocsparse_datatype value_type =
#if defined(QIWU_SPMM_FP64)
    rocsparse_datatype_f64_r;
#else
    rocsparse_datatype_f32_r;
#endif

void check(rocsparse_status status, const char* operation) {
    if (status != rocsparse_status_success) {
        throw std::runtime_error(
            std::string(operation) + " failed with rocSPARSE status " +
            std::to_string(static_cast<int>(status))
        );
    }
}

void validate(const QiwuSpmmProblem* problem, const QiwuSpmmExecutionContext* context) {
    if (!problem || !context || problem->data_type != QIWU_SPMM_DATA_TYPE ||
        problem->rows <= 0 || problem->cols <= 0 || problem->rhs_columns <= 0 ||
        problem->nnz <= 0 || problem->rows > std::numeric_limits<int32_t>::max() ||
        problem->cols > std::numeric_limits<int32_t>::max() ||
        problem->nnz > std::numeric_limits<int32_t>::max() ||
        problem->dense_layout != QiwuSpmmDenseLayout::row_major ||
        problem->op_a != QiwuSpmmOperation::none ||
        problem->op_b != QiwuSpmmOperation::none ||
        problem->ldb != problem->rhs_columns || problem->ldc != problem->rhs_columns) {
        throw std::runtime_error("unsupported rocSPARSE CSR SpMM problem");
    }
}

}  // namespace kernelperf_rocsparse_spmm

struct QiwuSpmmStorage {
    rocsparse_handle handle = nullptr;
    rocsparse_spmat_descr matrix = nullptr;
    rocsparse_dnmat_descr dense_b = nullptr;
    rocsparse_dnmat_descr dense_c = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    std::string version;
};

namespace kernelperf_rocsparse_spmm {

void release(QiwuSpmmStorage* storage) noexcept {
    if (!storage) return;
    if (storage->workspace) cudaFree(storage->workspace);
    if (storage->dense_c) rocsparse_destroy_dnmat_descr(storage->dense_c);
    if (storage->dense_b) rocsparse_destroy_dnmat_descr(storage->dense_b);
    if (storage->matrix) rocsparse_destroy_spmat_descr(storage->matrix);
    if (storage->handle) rocsparse_destroy_handle(storage->handle);
    delete storage;
}

}  // namespace kernelperf_rocsparse_spmm

extern "C" QiwuSpmmStorage* qiwu_spmm_preprocess(
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false) {
    using namespace kernelperf_rocsparse_spmm;
    validate(problem, context);
    auto* storage = new QiwuSpmmStorage();
    try {
        check(rocsparse_create_handle(&storage->handle), "rocsparse_create_handle");
        check(rocsparse_set_stream(storage->handle, stream), "rocsparse_set_stream");
        check(rocsparse_create_csr_descr(
            &storage->matrix,
            problem->rows,
            problem->cols,
            problem->nnz,
            const_cast<int32_t*>(problem->device_row_offsets),
            const_cast<int32_t*>(problem->device_column_indices),
            const_cast<QiwuSpmmScalar*>(problem->device_values),
            rocsparse_indextype_i32,
            rocsparse_indextype_i32,
            rocsparse_index_base_zero,
            value_type
        ), "rocsparse_create_csr_descr");
        check(rocsparse_create_dnmat_descr(
            &storage->dense_b,
            problem->cols,
            problem->rhs_columns,
            problem->ldb,
            const_cast<QiwuSpmmScalar*>(problem->device_b),
            value_type,
            rocsparse_order_row
        ), "rocsparse_create_dnmat_descr(B)");
        check(rocsparse_create_dnmat_descr(
            &storage->dense_c,
            problem->rows,
            problem->rhs_columns,
            problem->ldc,
            problem->device_c,
            value_type,
            rocsparse_order_row
        ), "rocsparse_create_dnmat_descr(C)");
        check(rocsparse_spmm(
            storage->handle,
            rocsparse_operation_none,
            rocsparse_operation_none,
            &context->alpha,
            storage->matrix,
            storage->dense_b,
            &context->beta,
            storage->dense_c,
            value_type,
            KERNELPERF_ROCSPARSE_SPMM_ALGORITHM,
            rocsparse_spmm_stage_buffer_size,
            &storage->workspace_size,
            nullptr
        ), "rocsparse_spmm(buffer_size)");
        if (storage->workspace_size > 0) {
            qiwu_spmm_check_gpu(
                cudaMalloc(&storage->workspace, storage->workspace_size),
                "allocate rocSPARSE SpMM workspace"
            );
        }
        check(rocsparse_spmm(
            storage->handle,
            rocsparse_operation_none,
            rocsparse_operation_none,
            &context->alpha,
            storage->matrix,
            storage->dense_b,
            &context->beta,
            storage->dense_c,
            value_type,
            KERNELPERF_ROCSPARSE_SPMM_ALGORITHM,
            rocsparse_spmm_stage_preprocess,
            &storage->workspace_size,
            storage->workspace
        ), "rocsparse_spmm(preprocess)");
        int version = 0;
        char revision[64] = {};
        check(rocsparse_get_version(storage->handle, &version), "rocsparse_get_version");
        check(rocsparse_get_git_rev(storage->handle, revision), "rocsparse_get_git_rev");
        char text[160];
        std::snprintf(
            text,
            sizeof(text),
            "rocSPARSE %d.%d.%d (%s)",
            version / 100000,
            version / 100 % 1000,
            version % 100,
            revision
        );
        storage->version = text;
        return storage;
    } catch (...) {
        cudaStreamSynchronize(stream);
        release(storage);
        throw;
    }
}

extern "C" void qiwu_spmm_solve(
    QiwuSpmmStorage* storage,
    const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext* context,
    QiwuSpmmStream stream
) noexcept(false) {
    using namespace kernelperf_rocsparse_spmm;
    if (!storage) throw std::runtime_error("invalid rocSPARSE SpMM storage");
    validate(problem, context);
    check(rocsparse_set_stream(storage->handle, stream), "rocsparse_set_stream");
    if (context->reset_output) {
        qiwu_spmm_check_gpu(
            cudaMemsetAsync(
                problem->device_c,
                0,
                sizeof(QiwuSpmmScalar) * static_cast<size_t>(
                    problem->rows * problem->rhs_columns
                ),
                stream
            ),
            "reset rocSPARSE SpMM output"
        );
    }
    size_t workspace_size = storage->workspace_size;
    check(rocsparse_spmm(
        storage->handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        &context->alpha,
        storage->matrix,
        storage->dense_b,
        &context->beta,
        storage->dense_c,
        value_type,
        KERNELPERF_ROCSPARSE_SPMM_ALGORITHM,
        rocsparse_spmm_stage_compute,
        &workspace_size,
        storage->workspace
    ), "rocsparse_spmm(compute)");
}

extern "C" const char* qiwu_spmm_library_version(const QiwuSpmmStorage* storage) noexcept {
    return storage ? storage->version.c_str() : "rocSPARSE unknown";
}

extern "C" const char* qiwu_spmm_algorithm(const QiwuSpmmStorage*) noexcept {
    return KERNELPERF_ROCSPARSE_SPMM_ALGORITHM_NAME;
}

extern "C" void qiwu_spmm_destroy(
    QiwuSpmmStorage* storage,
    QiwuSpmmStream stream
) noexcept(false) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    kernelperf_rocsparse_spmm::release(storage);
    qiwu_spmm_check_gpu(status, "synchronize before rocSPARSE SpMM destroy");
}

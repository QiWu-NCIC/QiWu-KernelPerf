#include <qiwu/spmm_plugin.cuh>

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

struct QiwuSpmmStorage {};

__global__ void fill_incorrect(QiwuSpmmScalar* output, int64_t size, QiwuSpmmScalar value) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) output[index] = value;
}

extern "C" QiwuSpmmStorage* qiwu_spmm_preprocess(
    const QiwuSpmmProblem*, const QiwuSpmmExecutionContext*, QiwuSpmmStream
) noexcept(false) {
    return new QiwuSpmmStorage();
}

extern "C" void qiwu_spmm_solve(
    QiwuSpmmStorage*, const QiwuSpmmProblem* problem,
    const QiwuSpmmExecutionContext*, QiwuSpmmStream stream
) noexcept(false) {
    const char* mode = std::getenv("QIWU_TEST_INVALID_MODE");
    QiwuSpmmScalar value = 0;
    if (mode && std::string(mode) == "nan") value = std::numeric_limits<QiwuSpmmScalar>::quiet_NaN();
    if (mode && std::string(mode) == "inf") value = std::numeric_limits<QiwuSpmmScalar>::infinity();
    if (mode && std::string(mode) == "runtime") throw std::runtime_error("intentional runtime failure");
    const int64_t size = problem->rows * problem->rhs_columns;
    fill_incorrect<<<static_cast<unsigned int>((size + 255) / 256), 256, 0, stream>>>(problem->device_c, size, value);
    qiwu_spmm_check_gpu(cudaGetLastError(), "incorrect fixture launch");
}

extern "C" const char* qiwu_spmm_library_version(const QiwuSpmmStorage*) noexcept {
    return "incorrect-test-fixture";
}

extern "C" const char* qiwu_spmm_algorithm(const QiwuSpmmStorage*) noexcept {
    return "incorrect-fill";
}

extern "C" void qiwu_spmm_destroy(QiwuSpmmStorage* storage, QiwuSpmmStream) noexcept(false) {
    delete storage;
}

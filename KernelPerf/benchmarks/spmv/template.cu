#include <cuda_runtime.h>
#include <cusparse.h>
#include <qiwu/spmv_plugin.cuh>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace kernelperf {

struct CsrMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t declared_nnz = 0;
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> column_indices;
    std::vector<QiwuSpmvScalar> values;
    bool complex_projected_to_real = false;
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int environment_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return static_cast<int>(parsed);
}

int64_t environment_int64(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        throw std::runtime_error(std::string(name) + " is required");
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return static_cast<int64_t>(parsed);
}

double environment_double(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (!end || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0) {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return parsed;
}

CsrMatrix load_matrix_market(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open matrix: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty Matrix Market file: " + path);
    }
    std::istringstream header_stream(line);
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    header_stream >> banner >> object >> format >> field >> symmetry;
    object = lowercase(object);
    format = lowercase(format);
    field = lowercase(field);
    symmetry = lowercase(symmetry);
    if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
        throw std::runtime_error("only Matrix Market coordinate matrices are supported");
    }
    if (field != "real" && field != "integer" && field != "pattern" && field != "complex") {
        throw std::runtime_error("unsupported Matrix Market field: " + field);
    }
    const bool mirrored =
        symmetry == "symmetric" || symmetry == "hermitian" || symmetry == "skew-symmetric";
    const bool skew = symmetry == "skew-symmetric";
    if (!mirrored && symmetry != "general") {
        throw std::runtime_error("unsupported Matrix Market symmetry: " + symmetry);
    }
    if (mirrored && object == "matrix" && format == "coordinate") {
        // Matrix Market stores symmetric matrices as square matrices. Rejecting
        // malformed rectangular input avoids writing transposed rows out of range.
    }

    do {
        if (!std::getline(input, line)) {
            throw std::runtime_error("missing Matrix Market dimensions");
        }
    } while (line.empty() || line[0] == '%');

    CsrMatrix matrix;
    matrix.complex_projected_to_real = field == "complex";
    std::istringstream dimensions(line);
    dimensions >> matrix.rows >> matrix.cols >> matrix.declared_nnz;
    if (
        !dimensions || matrix.rows <= 0 || matrix.cols <= 0 || matrix.declared_nnz <= 0 ||
        matrix.rows > std::numeric_limits<int32_t>::max() ||
        matrix.cols > std::numeric_limits<int32_t>::max()
    ) {
        throw std::runtime_error("invalid or unsupported Matrix Market dimensions");
    }

    using Entry = std::tuple<int32_t, int32_t, QiwuSpmvScalar>;
    std::vector<Entry> entries;
    entries.reserve(static_cast<size_t>(matrix.declared_nnz) * (mirrored ? 2U : 1U));
    int64_t loaded = 0;
    while (loaded < matrix.declared_nnz && std::getline(input, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream row_stream(line);
        int64_t row = 0;
        int64_t col = 0;
        double value = 1.0;
        row_stream >> row >> col;
        if (!row_stream) {
            throw std::runtime_error("invalid Matrix Market entry");
        }
        if (field != "pattern") {
            row_stream >> value;
            if (!row_stream) {
                throw std::runtime_error("invalid Matrix Market entry value");
            }
            if (field == "complex") {
                double imaginary_value = 0.0;
                row_stream >> imaginary_value;
                if (!row_stream) {
                    throw std::runtime_error("invalid Matrix Market complex entry value");
                }
            }
        }
        --row;
        --col;
        if (row < 0 || row >= matrix.rows || col < 0 || col >= matrix.cols) {
            throw std::runtime_error("Matrix Market index out of bounds");
        }
        if (mirrored && matrix.rows != matrix.cols) {
            throw std::runtime_error("symmetric Matrix Market input must be square");
        }
        if (!std::isfinite(value)) {
            throw std::runtime_error("Matrix Market contains a non-finite value");
        }
        const auto typed_value = static_cast<QiwuSpmvScalar>(value);
        entries.emplace_back(
            static_cast<int32_t>(row),
            static_cast<int32_t>(col),
            typed_value
        );
        if (mirrored && row != col) {
            entries.emplace_back(
                static_cast<int32_t>(col),
                static_cast<int32_t>(row),
                skew ? -typed_value : typed_value
            );
        }
        ++loaded;
    }
    if (loaded != matrix.declared_nnz) {
        throw std::runtime_error("Matrix Market file ended before all entries were read");
    }
    if (entries.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("expanded matrix nnz exceeds signed int32 CSR capacity");
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        return std::tie(std::get<0>(left), std::get<1>(left))
            < std::tie(std::get<0>(right), std::get<1>(right));
    });

    matrix.row_offsets.assign(static_cast<size_t>(matrix.rows) + 1, 0);
    matrix.column_indices.resize(entries.size());
    matrix.values.resize(entries.size());
    for (const auto& entry : entries) {
        ++matrix.row_offsets[static_cast<size_t>(std::get<0>(entry)) + 1];
    }
    for (int64_t row = 0; row < matrix.rows; ++row) {
        matrix.row_offsets[static_cast<size_t>(row) + 1] +=
            matrix.row_offsets[static_cast<size_t>(row)];
    }
    std::vector<int32_t> cursor = matrix.row_offsets;
    for (const auto& entry : entries) {
        const int32_t row = std::get<0>(entry);
        const int32_t position = cursor[static_cast<size_t>(row)]++;
        matrix.column_indices[static_cast<size_t>(position)] = std::get<1>(entry);
        matrix.values[static_cast<size_t>(position)] = std::get<2>(entry);
    }
    return matrix;
}

struct StandardSpmvResult {
    std::vector<QiwuSpmvScalar> values;
    std::vector<long double> scales;
};

StandardSpmvResult standard_spmv(
    const CsrMatrix& matrix,
    const std::vector<QiwuSpmvScalar>& x
) {
    StandardSpmvResult result{
        std::vector<QiwuSpmvScalar>(static_cast<size_t>(matrix.rows), 0),
        std::vector<long double>(static_cast<size_t>(matrix.rows), 0),
    };
    std::vector<long double> accum(static_cast<size_t>(matrix.rows), 0);
    for (int64_t row = 0; row < matrix.rows; ++row) {
        const int32_t begin = matrix.row_offsets[static_cast<size_t>(row)];
        const int32_t end = matrix.row_offsets[static_cast<size_t>(row) + 1];
        for (int32_t index = begin; index < end; ++index) {
            const long double product =
                static_cast<long double>(matrix.values[static_cast<size_t>(index)]) *
                static_cast<long double>(x[static_cast<size_t>(matrix.column_indices[static_cast<size_t>(index)])]);
            accum[static_cast<size_t>(row)] += product;
            result.scales[static_cast<size_t>(row)] += std::abs(product);
        }
    }
    for (size_t row = 0; row < accum.size(); ++row) {
        result.values[row] = static_cast<QiwuSpmvScalar>(accum[row]);
    }
    return result;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        qiwu_spmv_check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
            "cudaMalloc platform buffer"
        );
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    T* get() const { return data_; }
    T* release() {
        T* value = data_;
        data_ = nullptr;
        return value;
    }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

class Stream {
public:
    Stream() {
        qiwu_spmv_check_cuda(
            cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags"
        );
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        if (value_) {
            cudaStreamDestroy(value_);
        }
    }
    operator cudaStream_t() const { return value_; }

private:
    cudaStream_t value_ = nullptr;
};

class Event {
public:
    Event() { qiwu_spmv_check_cuda(cudaEventCreate(&value_), "cudaEventCreate"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (value_) {
            cudaEventDestroy(value_);
        }
    }
    operator cudaEvent_t() const { return value_; }

private:
    cudaEvent_t value_ = nullptr;
};

class CandidateStorageGuard {
public:
    CandidateStorageGuard(
        QiwuSpmvStorage*& storage,
        cudaStream_t stream,
        std::string& destroy_error
    ) : storage_(storage), stream_(stream), destroy_error_(destroy_error) {}

    CandidateStorageGuard(const CandidateStorageGuard&) = delete;
    CandidateStorageGuard& operator=(const CandidateStorageGuard&) = delete;

    ~CandidateStorageGuard() noexcept {
        if (!storage_) {
            return;
        }
        QiwuSpmvStorage* failed_storage = storage_;
        storage_ = nullptr;
        try {
            qiwu_spmv_destroy(failed_storage, stream_);
        } catch (const std::exception& error) {
            destroy_error_ = error.what();
            cudaDeviceSynchronize();
        } catch (...) {
            destroy_error_ = "unknown exception";
            cudaDeviceSynchronize();
        }
    }

private:
    QiwuSpmvStorage*& storage_;
    cudaStream_t stream_;
    std::string& destroy_error_;
};

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

void emit_failure(
    const std::string& stage,
    const std::string& error,
    const std::string& destroy_error
) {
    std::cout << "{\"status\":\"error\",\"stage\":\"" << json_escape(stage)
              << "\",\"error\":\"" << json_escape(error) << "\"";
    if (!destroy_error.empty()) {
        std::cout << ",\"destroy_error\":\"" << json_escape(destroy_error) << "\"";
    }
    std::cout << "}" << std::endl;
}

const char* data_type_name() {
#if defined(QIWU_SPMV_FP64)
    return "fp64";
#else
    return "fp32";
#endif
}

}  // namespace kernelperf

int main() {
    using namespace kernelperf;

    QiwuSpmvStorage* storage = nullptr;
    std::string stage = "setup";
    std::string destroy_error;
    try {
        const char* matrix_path = std::getenv("KERNELPERF_MATRIX_PATH");
        if (!matrix_path || !*matrix_path) {
            throw std::runtime_error("KERNELPERF_MATRIX_PATH is required");
        }
        const int warmup = environment_int("KERNELPERF_WARMUP", 5);
        const int iterations = environment_int("KERNELPERF_ITERATIONS", 20);
        const int64_t expected_rows = environment_int64("KERNELPERF_MATRIX_ROWS");
        const int64_t expected_cols = environment_int64("KERNELPERF_MATRIX_COLS");
        const double validation_tolerance = environment_double(
            "KERNELPERF_VALIDATION_TOLERANCE",
#if defined(QIWU_SPMV_FP64)
            1e-12
#else
            1e-4
#endif
        );

        const CsrMatrix matrix = load_matrix_market(matrix_path);
        if (matrix.rows != expected_rows || matrix.cols != expected_cols) {
            std::ostringstream message;
            message << "dataset metadata mismatch: expected " << expected_rows << "x"
                    << expected_cols << ", loaded " << matrix.rows << "x" << matrix.cols;
            throw std::runtime_error(message.str());
        }

        std::vector<QiwuSpmvScalar> host_x(static_cast<size_t>(matrix.cols));
        for (int64_t col = 0; col < matrix.cols; ++col) {
            const int value = static_cast<int>(
                (static_cast<uint64_t>(col) * 17U + 13U) % 31U
            ) - 15;
            host_x[static_cast<size_t>(col)] =
                static_cast<QiwuSpmvScalar>(value) /
                static_cast<QiwuSpmvScalar>(7);
        }
        const StandardSpmvResult expected = standard_spmv(matrix, host_x);
        const std::vector<QiwuSpmvScalar> sentinel(
            static_cast<size_t>(matrix.rows),
            std::numeric_limits<QiwuSpmvScalar>::quiet_NaN()
        );

        Stream stream;
        DeviceBuffer<int32_t> device_row_offsets(matrix.row_offsets.size());
        DeviceBuffer<int32_t> device_column_indices(matrix.column_indices.size());
        DeviceBuffer<QiwuSpmvScalar> device_values(matrix.values.size());
        DeviceBuffer<QiwuSpmvScalar> device_x(host_x.size());
        DeviceBuffer<QiwuSpmvScalar> device_y(sentinel.size());
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                device_row_offsets.get(),
                matrix.row_offsets.data(),
                device_row_offsets.bytes(),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy CSR row offsets"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                device_column_indices.get(),
                matrix.column_indices.data(),
                device_column_indices.bytes(),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy CSR column indices"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                device_values.get(),
                matrix.values.data(),
                device_values.bytes(),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy CSR values"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                device_x.get(), host_x.data(), device_x.bytes(), cudaMemcpyHostToDevice, stream
            ),
            "copy x"
        );
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                device_y.get(), sentinel.data(), device_y.bytes(), cudaMemcpyHostToDevice, stream
            ),
            "initialize y"
        );
        qiwu_spmv_check_cuda(
            cudaStreamSynchronize(stream),
            "synchronize platform input setup"
        );

        const QiwuSpmvCsrInput input{
            QIWU_SPMV_DATA_TYPE,
            matrix.rows,
            matrix.cols,
            static_cast<int64_t>(matrix.values.size()),
            matrix.row_offsets.data(),
            matrix.column_indices.data(),
            matrix.values.data(),
            device_row_offsets.get(),
            device_column_indices.get(),
            device_values.get(),
        };
        const QiwuSpmvExecutionContext context{
            device_x.get(),
            device_y.get(),
        };
        CandidateStorageGuard storage_guard(storage, stream, destroy_error);

        stage = "preprocess";
        const auto preprocess_start = std::chrono::steady_clock::now();
        storage = qiwu_spmv_preprocess(&input, &context, stream);
        if (!storage) {
            throw std::runtime_error("qiwu_spmv_preprocess returned null");
        }
        qiwu_spmv_check_cuda(
            cudaStreamSynchronize(stream),
            "synchronize candidate preprocess"
        );
        const auto preprocess_stop = std::chrono::steady_clock::now();
        const double preprocess_ms = std::chrono::duration<double, std::milli>(
            preprocess_stop - preprocess_start
        ).count();

        stage = "warmup";
        for (int iteration = 0; iteration < warmup; ++iteration) {
            qiwu_spmv_solve(storage, &context, stream);
        }
        qiwu_spmv_check_cuda(cudaGetLastError(), "warmup launch");
        qiwu_spmv_check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

        stage = "solve";
#if defined(KERNELPERF_SPMV_HOST_TIMING)
        const auto solve_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            qiwu_spmv_solve(storage, &context, stream);
        }
        qiwu_spmv_check_cuda(cudaGetLastError(), "timed solve launch");
        qiwu_spmv_check_cuda(cudaDeviceSynchronize(), "synchronize host-timed solve");
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - solve_start
        ).count();
#else
        Event start;
        Event stop;
        qiwu_spmv_check_cuda(cudaEventRecord(start, stream), "record solve start");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            qiwu_spmv_solve(storage, &context, stream);
        }
        qiwu_spmv_check_cuda(cudaGetLastError(), "timed solve launch");
        qiwu_spmv_check_cuda(cudaEventRecord(stop, stream), "record solve stop");
        qiwu_spmv_check_cuda(cudaEventSynchronize(stop), "synchronize solve stop");
        float elapsed_ms = 0.0f;
        qiwu_spmv_check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "measure solve elapsed time"
        );
#endif

        stage = "validate";
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                device_y.get(), sentinel.data(), device_y.bytes(), cudaMemcpyHostToDevice, stream
            ),
            "reset y sentinel"
        );
        qiwu_spmv_solve(storage, &context, stream);
        qiwu_spmv_check_cuda(cudaGetLastError(), "validation launch");
        std::vector<QiwuSpmvScalar> actual(static_cast<size_t>(matrix.rows));
        qiwu_spmv_check_cuda(
            cudaMemcpyAsync(
                actual.data(), device_y.get(), device_y.bytes(), cudaMemcpyDeviceToHost, stream
            ),
            "copy validation y"
        );
        qiwu_spmv_check_cuda(
            cudaStreamSynchronize(stream),
            "validation synchronize"
        );

        long double max_absolute_error = 0;
        long double max_relative_error = 0;
        long double max_normalized_error = 0;
        size_t mismatch_count = 0;
        for (int64_t row = 0; row < matrix.rows; ++row) {
            const long double actual_value =
                static_cast<long double>(actual[static_cast<size_t>(row)]);
            const long double expected_value =
                static_cast<long double>(expected.values[static_cast<size_t>(row)]);
            if (!std::isfinite(actual_value)) {
                ++mismatch_count;
                continue;
            }
            const long double absolute_error = std::abs(actual_value - expected_value);
            const long double relative_error = absolute_error /
                std::max(std::abs(expected_value), static_cast<long double>(1));
            const long double normalized_error = absolute_error /
                std::max(
                    expected.scales[static_cast<size_t>(row)],
                    static_cast<long double>(1)
                );
            max_absolute_error = std::max(max_absolute_error, absolute_error);
            max_relative_error = std::max(max_relative_error, relative_error);
            max_normalized_error = std::max(max_normalized_error, normalized_error);
            if (normalized_error > validation_tolerance) {
                ++mismatch_count;
            }
        }

        stage = "destroy";
        QiwuSpmvStorage* completed_storage = storage;
        storage = nullptr;
        try {
            qiwu_spmv_destroy(completed_storage, stream);
            qiwu_spmv_check_cuda(
                cudaStreamSynchronize(stream),
                "synchronize candidate destroy"
            );
        } catch (...) {
            device_row_offsets.release();
            device_column_indices.release();
            device_values.release();
            device_x.release();
            device_y.release();
            throw;
        }

        const double runtime_ms = static_cast<double>(elapsed_ms) / iterations;
        if (!std::isfinite(runtime_ms) || runtime_ms <= 0.0) {
            throw std::runtime_error("solve timing produced a non-positive runtime");
        }
        const double operations = 2.0 * static_cast<double>(matrix.values.size());
        const bool valid = mismatch_count == 0;
        int runtime_version = 0;
        int driver_version = 0;
        int cusparse_major = 0;
        int cusparse_minor = 0;
        int cusparse_patch = 0;
        cudaRuntimeGetVersion(&runtime_version);
        cudaDriverGetVersion(&driver_version);
        cusparseGetProperty(MAJOR_VERSION, &cusparse_major);
        cusparseGetProperty(MINOR_VERSION, &cusparse_minor);
        cusparseGetProperty(PATCH_LEVEL, &cusparse_patch);
        std::cout << std::setprecision(12)
                  << "{\"status\":\"ok\""
                  << ",\"preprocess_ms\":" << preprocess_ms
                  << ",\"runtime_ms\":" << runtime_ms
                  << ",\"operations\":" << operations
                  << ",\"valid\":" << (valid ? "true" : "false")
                  << ",\"metadata\":{\"dtype\":\"" << data_type_name() << "\""
                  << ",\"index_type\":\"int32\""
                  << ",\"actual_nnz\":" << matrix.values.size()
                  << ",\"declared_nnz\":" << matrix.declared_nnz
                  << ",\"complex_projected_to_real\":"
                  << (matrix.complex_projected_to_real ? "true" : "false")
                  << ",\"max_absolute_error\":" << static_cast<double>(max_absolute_error)
                  << ",\"max_relative_error\":" << static_cast<double>(max_relative_error)
                  << ",\"max_normalized_error\":" << static_cast<double>(max_normalized_error)
                  << ",\"mismatch_count\":" << mismatch_count
                  << ",\"validation_tolerance\":" << validation_tolerance
                  << ",\"reference_precision\":\"" << data_type_name() << "\""
                  << ",\"warmup\":" << warmup
                  << ",\"iterations\":" << iterations
                  << ",\"cuda_runtime_version\":" << runtime_version
                  << ",\"cuda_driver_version\":" << driver_version
                  << ",\"cusparse_version\":\"" << cusparse_major << "."
                  << cusparse_minor << "." << cusparse_patch << "\""
                  << "}}" << std::endl;
#if defined(KERNELPERF_GHOST_CUDA)
        // GHOST's CUDA 12 teardown registers a second ownership path for its
        // global task/map state.  Each benchmark is a single-case process;
        // quick_exit preserves the flushed measurement while bypassing the
        // duplicate static cleanup that otherwise aborts with double-free.
        std::quick_exit(0);
#endif
        return 0;
    } catch (const std::exception& error) {
        const std::string primary_error = error.what();
        emit_failure(stage, primary_error, destroy_error);
        return 1;
    } catch (...) {
        emit_failure(stage, "unknown exception", destroy_error);
        return 1;
    }
}

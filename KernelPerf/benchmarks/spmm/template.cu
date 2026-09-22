#include <qiwu/gpu_runtime.h>
#if !defined(QIWU_BACKEND_HIP)
#include <cusparse.h>
#endif
#include <qiwu/spmm_plugin.cuh>

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
    std::vector<QiwuSpmmScalar> values;
    bool complex_projected_to_real = false;
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

int environment_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return static_cast<int>(parsed);
}

int64_t environment_int64(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) throw std::runtime_error(std::string(name) + " is required");
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return static_cast<int64_t>(parsed);
}

std::vector<int64_t> environment_int64_list(const char* name) {
    const char* raw_value = std::getenv(name);
    if (!raw_value || !*raw_value) throw std::runtime_error(std::string(name) + " is required");
    const std::string raw(raw_value);
    std::vector<int64_t> values;
    std::istringstream input(raw);
    std::string token;
    while (std::getline(input, token, ',')) {
        char* end = nullptr;
        const long long parsed = std::strtoll(token.c_str(), &end, 10);
        if (!end || *end != '\0' || parsed <= 0) {
            throw std::runtime_error(std::string("invalid environment value: ") + name);
        }
        const int64_t value = static_cast<int64_t>(parsed);
        if (std::find(values.begin(), values.end(), value) != values.end()) {
            throw std::runtime_error(std::string("duplicate environment value: ") + name);
        }
        values.push_back(value);
    }
    return values;
}

uint64_t environment_uint64(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0') {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return static_cast<uint64_t>(parsed);
}

double environment_double(const char* name, double fallback, bool allow_zero = false) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (!end || *end != '\0' || !std::isfinite(parsed) || (allow_zero ? parsed < 0.0 : parsed <= 0.0)) {
        throw std::runtime_error(std::string("invalid environment value: ") + name);
    }
    return parsed;
}

std::string environment_string(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : std::string(fallback);
}

CsrMatrix load_matrix_market(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open matrix: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty Matrix Market file: " + path);
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
    const bool mirrored = symmetry == "symmetric" || symmetry == "hermitian" || symmetry == "skew-symmetric";
    const bool skew = symmetry == "skew-symmetric";
    if (!mirrored && symmetry != "general") {
        throw std::runtime_error("unsupported Matrix Market symmetry: " + symmetry);
    }
    do {
        if (!std::getline(input, line)) throw std::runtime_error("missing Matrix Market dimensions");
    } while (line.empty() || line[0] == '%');

    CsrMatrix matrix;
    matrix.complex_projected_to_real = field == "complex";
    std::istringstream dimensions(line);
    dimensions >> matrix.rows >> matrix.cols >> matrix.declared_nnz;
    if (!dimensions || matrix.rows <= 0 || matrix.cols <= 0 || matrix.declared_nnz <= 0 ||
        matrix.rows > std::numeric_limits<int32_t>::max() ||
        matrix.cols > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("invalid or unsupported Matrix Market dimensions");
    }
    if (mirrored && matrix.rows != matrix.cols) {
        throw std::runtime_error("symmetric Matrix Market input must be square");
    }

    using Entry = std::tuple<int32_t, int32_t, QiwuSpmmScalar>;
    std::vector<Entry> entries;
    entries.reserve(static_cast<size_t>(matrix.declared_nnz) * (mirrored ? 2U : 1U));
    int64_t loaded = 0;
    while (loaded < matrix.declared_nnz && std::getline(input, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream row_stream(line);
        int64_t row = 0;
        int64_t column = 0;
        double value = 1.0;
        row_stream >> row >> column;
        if (!row_stream) throw std::runtime_error("invalid Matrix Market entry");
        if (field != "pattern") {
            row_stream >> value;
            if (!row_stream) throw std::runtime_error("invalid Matrix Market entry value");
            if (field == "complex") {
                double imaginary = 0.0;
                row_stream >> imaginary;
                if (!row_stream) throw std::runtime_error("invalid Matrix Market complex entry value");
            }
        }
        --row;
        --column;
        if (row < 0 || row >= matrix.rows || column < 0 || column >= matrix.cols) {
            throw std::runtime_error("Matrix Market index out of bounds");
        }
        if (!std::isfinite(value)) throw std::runtime_error("Matrix Market contains a non-finite value");
        const auto typed_value = static_cast<QiwuSpmmScalar>(value);
        entries.emplace_back(static_cast<int32_t>(row), static_cast<int32_t>(column), typed_value);
        if (mirrored && row != column) {
            entries.emplace_back(
                static_cast<int32_t>(column),
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
        return std::tie(std::get<0>(left), std::get<1>(left)) <
            std::tie(std::get<0>(right), std::get<1>(right));
    });
    matrix.row_offsets.assign(static_cast<size_t>(matrix.rows) + 1, 0);
    matrix.column_indices.resize(entries.size());
    matrix.values.resize(entries.size());
    for (const auto& entry : entries) {
        ++matrix.row_offsets[static_cast<size_t>(std::get<0>(entry)) + 1];
    }
    for (int64_t row = 0; row < matrix.rows; ++row) {
        matrix.row_offsets[static_cast<size_t>(row) + 1] += matrix.row_offsets[static_cast<size_t>(row)];
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

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::vector<QiwuSpmmScalar> make_dense_b(int64_t rows, int64_t columns, uint64_t seed) {
    std::vector<QiwuSpmmScalar> values(static_cast<size_t>(rows * columns));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
            const uint64_t mixed = splitmix64(
                seed ^ (static_cast<uint64_t>(row) * 0x9e3779b97f4a7c15ULL) ^
                (static_cast<uint64_t>(column) * 0xd1b54a32d192ed03ULL)
            );
            const int32_t centered = static_cast<int32_t>(mixed & 0xffffU) - 32768;
            values[static_cast<size_t>(row * columns + column)] =
                static_cast<QiwuSpmmScalar>(centered) / static_cast<QiwuSpmmScalar>(8192);
        }
    }
    return values;
}

struct ReferenceResult {
    std::vector<long double> values;
    std::vector<long double> scales;
};

ReferenceResult standard_spmm(
    const CsrMatrix& matrix,
    const std::vector<QiwuSpmmScalar>& dense_b,
    int64_t rhs_columns
) {
    const size_t output_size = static_cast<size_t>(matrix.rows * rhs_columns);
    ReferenceResult result{
        std::vector<long double>(output_size, 0.0L),
        std::vector<long double>(output_size, 0.0L),
    };
    for (int64_t row = 0; row < matrix.rows; ++row) {
        const int32_t begin = matrix.row_offsets[static_cast<size_t>(row)];
        const int32_t end = matrix.row_offsets[static_cast<size_t>(row) + 1];
        for (int32_t index = begin; index < end; ++index) {
            const int32_t sparse_column = matrix.column_indices[static_cast<size_t>(index)];
            const long double matrix_value = static_cast<long double>(matrix.values[static_cast<size_t>(index)]);
            for (int64_t rhs = 0; rhs < rhs_columns; ++rhs) {
                const size_t output_index = static_cast<size_t>(row * rhs_columns + rhs);
                const long double product = matrix_value * static_cast<long double>(
                    dense_b[static_cast<size_t>(sparse_column * rhs_columns + rhs)]
                );
                result.values[output_index] += product;
                result.scales[output_index] += std::abs(product);
            }
        }
    }
    return result;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        qiwu_spmm_check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
            "cudaMalloc platform buffer"
        );
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    T* get() const { return data_; }
    T* release() { T* value = data_; data_ = nullptr; return value; }
    size_t bytes() const { return count_ * sizeof(T); }
private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

class Stream {
public:
    Stream() {
        qiwu_spmm_check_cuda(
            cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags"
        );
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() { if (value_) cudaStreamDestroy(value_); }
    operator cudaStream_t() const { return value_; }
private:
    cudaStream_t value_ = nullptr;
};

class Event {
public:
    Event() { qiwu_spmm_check_cuda(cudaEventCreate(&value_), "cudaEventCreate"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { if (value_) cudaEventDestroy(value_); }
    operator cudaEvent_t() const { return value_; }
private:
    cudaEvent_t value_ = nullptr;
};

class CandidateStorageGuard {
public:
    CandidateStorageGuard(QiwuSpmmStorage*& storage, cudaStream_t stream, std::string& destroy_error)
        : storage_(storage), stream_(stream), destroy_error_(destroy_error) {}
    CandidateStorageGuard(const CandidateStorageGuard&) = delete;
    CandidateStorageGuard& operator=(const CandidateStorageGuard&) = delete;
    ~CandidateStorageGuard() noexcept {
        if (!storage_) return;
        QiwuSpmmStorage* failed_storage = storage_;
        storage_ = nullptr;
        try {
            qiwu_spmm_destroy(failed_storage, stream_);
        } catch (const std::exception& error) {
            destroy_error_ = error.what();
            cudaDeviceSynchronize();
        } catch (...) {
            destroy_error_ = "unknown exception";
            cudaDeviceSynchronize();
        }
    }
private:
    QiwuSpmmStorage*& storage_;
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
    const std::string& destroy_error,
    int64_t rhs_columns = 0
) {
    std::cout << "{\"status\":\"error\",\"stage\":\"" << json_escape(stage)
              << "\",\"error\":\"" << json_escape(error) << "\"";
    if (rhs_columns > 0) std::cout << ",\"rhs_columns\":" << rhs_columns;
    if (!destroy_error.empty()) {
        std::cout << ",\"destroy_error\":\"" << json_escape(destroy_error) << "\"";
    }
    std::cout << "}" << std::endl;
}

const char* data_type_name() {
#if defined(QIWU_SPMM_FP64)
    return "fp64";
#else
    return "fp32";
#endif
}

void write_json_number(std::ostream& output, long double value) {
    const double narrowed = static_cast<double>(value);
    if (std::isfinite(narrowed)) output << narrowed;
    else output << "null";
}

}  // namespace kernelperf

int main() {
    using namespace kernelperf;
    QiwuSpmmStorage* storage = nullptr;
    std::string stage = "setup";
    std::string destroy_error;
    try {
        const char* matrix_path = std::getenv("KERNELPERF_MATRIX_PATH");
        if (!matrix_path || !*matrix_path) throw std::runtime_error("KERNELPERF_MATRIX_PATH is required");
        const int warmup = environment_int("KERNELPERF_WARMUP", 5);
        const int iterations = environment_int("KERNELPERF_ITERATIONS", 20);
        const int64_t expected_rows = environment_int64("KERNELPERF_MATRIX_ROWS");
        const int64_t expected_cols = environment_int64("KERNELPERF_MATRIX_COLS");
        const std::vector<int64_t> rhs_values = environment_int64_list("KERNELPERF_RHS_COLUMNS");
        const uint64_t input_seed = environment_uint64("KERNELPERF_INPUT_SEED", 20260922ULL);
        const double validation_safety_factor = environment_double(
            "KERNELPERF_VALIDATION_SAFETY_FACTOR", 4.0
        );
        const double alpha_value = environment_double("KERNELPERF_ALPHA", 1.0);
        const double beta_value = environment_double("KERNELPERF_BETA", 0.0, true);
        const std::string dense_layout = lowercase(environment_string("KERNELPERF_DENSE_LAYOUT", "row-major"));
        const std::string op_a = lowercase(environment_string("KERNELPERF_OP_A", "N"));
        const std::string op_b = lowercase(environment_string("KERNELPERF_OP_B", "N"));
        if (dense_layout != "row-major" || op_a != "n" || op_b != "n") {
            throw std::runtime_error("P0 SpMM requires row-major B/C with op(A)=N and op(B)=N");
        }
        if (alpha_value != 1.0 || beta_value != 0.0) {
            throw std::runtime_error("P0 SpMM requires alpha=1 and beta=0");
        }

        const CsrMatrix matrix = load_matrix_market(matrix_path);
        if (matrix.rows != expected_rows || matrix.cols != expected_cols) {
            std::ostringstream message;
            message << "dataset metadata mismatch: expected " << expected_rows << "x" << expected_cols
                    << ", loaded " << matrix.rows << "x" << matrix.cols;
            throw std::runtime_error(message.str());
        }
        for (const int64_t rhs_columns : rhs_values) {
            storage = nullptr;
            stage = "setup";
            destroy_error.clear();
            try {
        const std::vector<QiwuSpmmScalar> host_b = make_dense_b(matrix.cols, rhs_columns, input_seed);
        const ReferenceResult expected = standard_spmm(matrix, host_b, rhs_columns);
        const std::vector<QiwuSpmmScalar> sentinel(
            static_cast<size_t>(matrix.rows * rhs_columns),
            std::numeric_limits<QiwuSpmmScalar>::quiet_NaN()
        );

        Stream stream;
        DeviceBuffer<int32_t> device_row_offsets(matrix.row_offsets.size());
        DeviceBuffer<int32_t> device_column_indices(matrix.column_indices.size());
        DeviceBuffer<QiwuSpmmScalar> device_values(matrix.values.size());
        DeviceBuffer<QiwuSpmmScalar> device_b(host_b.size());
        DeviceBuffer<QiwuSpmmScalar> device_c(sentinel.size());
        qiwu_spmm_check_cuda(cudaMemcpyAsync(device_row_offsets.get(), matrix.row_offsets.data(),
            device_row_offsets.bytes(), cudaMemcpyHostToDevice, stream), "copy CSR row offsets");
        qiwu_spmm_check_cuda(cudaMemcpyAsync(device_column_indices.get(), matrix.column_indices.data(),
            device_column_indices.bytes(), cudaMemcpyHostToDevice, stream), "copy CSR column indices");
        qiwu_spmm_check_cuda(cudaMemcpyAsync(device_values.get(), matrix.values.data(),
            device_values.bytes(), cudaMemcpyHostToDevice, stream), "copy CSR values");
        qiwu_spmm_check_cuda(cudaMemcpyAsync(device_b.get(), host_b.data(), device_b.bytes(),
            cudaMemcpyHostToDevice, stream), "copy dense B");
        qiwu_spmm_check_cuda(cudaMemcpyAsync(device_c.get(), sentinel.data(), device_c.bytes(),
            cudaMemcpyHostToDevice, stream), "initialize dense C");
        qiwu_spmm_check_cuda(cudaStreamSynchronize(stream), "synchronize platform input setup");

        const QiwuSpmmProblem problem{
            QIWU_SPMM_DATA_TYPE,
            matrix.rows,
            matrix.cols,
            rhs_columns,
            static_cast<int64_t>(matrix.values.size()),
            matrix.row_offsets.data(),
            matrix.column_indices.data(),
            matrix.values.data(),
            device_row_offsets.get(),
            device_column_indices.get(),
            device_values.get(),
            device_b.get(),
            device_c.get(),
            rhs_columns,
            rhs_columns,
            QiwuSpmmDenseLayout::row_major,
            QiwuSpmmOperation::none,
            QiwuSpmmOperation::none,
        };
        QiwuSpmmExecutionContext context{
            static_cast<QiwuSpmmScalar>(alpha_value),
            static_cast<QiwuSpmmScalar>(beta_value),
        };
        CandidateStorageGuard storage_guard(storage, stream, destroy_error);

        stage = "preprocess";
        const auto preprocess_start = std::chrono::steady_clock::now();
        storage = qiwu_spmm_preprocess(&problem, &context, stream);
        if (!storage) throw std::runtime_error("qiwu_spmm_preprocess returned null");
        qiwu_spmm_check_cuda(cudaStreamSynchronize(stream), "synchronize candidate preprocess");
        const double preprocess_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preprocess_start
        ).count();

        stage = "warmup";
        for (int iteration = 0; iteration < warmup; ++iteration) {
            qiwu_spmm_solve(storage, &problem, &context, stream);
        }
        qiwu_spmm_check_cuda(cudaGetLastError(), "warmup launch");
        qiwu_spmm_check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

        stage = "solve";
#if defined(KERNELPERF_SPMM_HOST_TIMING)
        const auto solve_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            qiwu_spmm_solve(storage, &problem, &context, stream);
        }
        qiwu_spmm_check_cuda(cudaGetLastError(), "timed solve launch");
        qiwu_spmm_check_cuda(cudaDeviceSynchronize(), "synchronize host-timed solve");
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - solve_start
        ).count();
#else
        Event start;
        Event stop;
        qiwu_spmm_check_cuda(cudaEventRecord(start, stream), "record solve start");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            qiwu_spmm_solve(storage, &problem, &context, stream);
        }
        qiwu_spmm_check_cuda(cudaGetLastError(), "timed solve launch");
        qiwu_spmm_check_cuda(cudaEventRecord(stop, stream), "record solve stop");
        qiwu_spmm_check_cuda(cudaEventSynchronize(stop), "synchronize solve stop");
        float elapsed_ms = 0.0f;
        qiwu_spmm_check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "measure solve elapsed time");
#endif

        stage = "validate";
        qiwu_spmm_check_cuda(cudaMemcpyAsync(device_c.get(), sentinel.data(), device_c.bytes(),
            cudaMemcpyHostToDevice, stream), "reset C sentinel");
        context.reset_output = true;
        qiwu_spmm_solve(storage, &problem, &context, stream);
        context.reset_output = false;
        qiwu_spmm_check_cuda(cudaGetLastError(), "validation launch");
        std::vector<QiwuSpmmScalar> actual(sentinel.size());
        qiwu_spmm_check_cuda(cudaMemcpyAsync(actual.data(), device_c.get(), device_c.bytes(),
            cudaMemcpyDeviceToHost, stream), "copy validation C");
        qiwu_spmm_check_cuda(cudaStreamSynchronize(stream), "validation synchronize");

        long double max_absolute_error = 0.0L;
        long double max_relative_error = 0.0L;
        long double max_normalized_error = 0.0L;
        size_t failed_elements = 0;
        size_t invalid_elements = 0;
        size_t nan_elements = 0;
        size_t inf_elements = 0;
        int64_t max_row_nnz = 0;
        const long double unit_roundoff = static_cast<long double>(
            std::numeric_limits<QiwuSpmmScalar>::epsilon()
        ) / 2.0L;
        for (int64_t row = 0; row < matrix.rows; ++row) {
            const int32_t begin = matrix.row_offsets[static_cast<size_t>(row)];
            const int32_t end = matrix.row_offsets[static_cast<size_t>(row) + 1];
            const int64_t row_nnz = static_cast<int64_t>(end) - begin;
            max_row_nnz = std::max(max_row_nnz, row_nnz);
            const long double row_bound = validation_safety_factor *
                static_cast<long double>(row_nnz) * unit_roundoff;
            for (int64_t rhs = 0; rhs < rhs_columns; ++rhs) {
                const size_t index = static_cast<size_t>(row * rhs_columns + rhs);
                const QiwuSpmmScalar stored = actual[index];
                if (std::isnan(stored)) {
                    ++nan_elements;
                    ++invalid_elements;
                    continue;
                }
                if (std::isinf(stored)) {
                    ++inf_elements;
                    ++invalid_elements;
                    continue;
                }
                const long double actual_value = static_cast<long double>(stored);
                const long double reference = expected.values[index];
                const long double absolute_error = std::abs(actual_value - reference);
                const long double scale = expected.scales[index];
                const long double relative_error = absolute_error /
                    std::max(std::abs(reference), std::numeric_limits<long double>::min());
                const long double normalized_error = scale > 0.0L
                    ? absolute_error / scale
                    : (absolute_error == 0.0L ? 0.0L : std::numeric_limits<long double>::infinity());
                max_absolute_error = std::max(max_absolute_error, absolute_error);
                max_relative_error = std::max(max_relative_error, relative_error);
                max_normalized_error = std::max(max_normalized_error, normalized_error);
                if ((scale == 0.0L && absolute_error != 0.0L) ||
                    (scale > 0.0L && normalized_error > row_bound)) {
                    ++failed_elements;
                }
            }
        }

        stage = "destroy";
        const char* library_version_pointer = qiwu_spmm_library_version(storage);
        const char* algorithm_pointer = qiwu_spmm_algorithm(storage);
        const std::string library_version = library_version_pointer ? library_version_pointer : "unknown";
        const std::string algorithm = algorithm_pointer ? algorithm_pointer : "unknown";
        QiwuSpmmStorage* completed_storage = storage;
        storage = nullptr;
        try {
            qiwu_spmm_destroy(completed_storage, stream);
            qiwu_spmm_check_cuda(cudaStreamSynchronize(stream), "synchronize candidate destroy");
        } catch (...) {
            device_row_offsets.release();
            device_column_indices.release();
            device_values.release();
            device_b.release();
            device_c.release();
            throw;
        }

        const double runtime_ms = static_cast<double>(elapsed_ms) / iterations;
        if (!std::isfinite(runtime_ms) || runtime_ms <= 0.0) {
            throw std::runtime_error("solve timing produced a non-positive runtime");
        }
        const double pre_plus_solve_ms = preprocess_ms + runtime_ms;
        const double pre_amortized_ms = preprocess_ms / iterations + runtime_ms;
        const double operations = 2.0 * static_cast<double>(matrix.values.size()) *
            static_cast<double>(rhs_columns);
        const bool valid = failed_elements == 0 && invalid_elements == 0;
        const char* validation_status = invalid_elements > 0
            ? "invalid-output"
            : (failed_elements > 0 ? "precision-error" : "pass");
        int runtime_version = 0;
        int driver_version = 0;
        int cusparse_major = 0;
        int cusparse_minor = 0;
        int cusparse_patch = 0;
        cudaRuntimeGetVersion(&runtime_version);
        cudaDriverGetVersion(&driver_version);
#if !defined(QIWU_BACKEND_HIP)
        cusparseGetProperty(MAJOR_VERSION, &cusparse_major);
        cusparseGetProperty(MINOR_VERSION, &cusparse_minor);
        cusparseGetProperty(PATCH_LEVEL, &cusparse_patch);
#endif
        std::cout << std::setprecision(12)
                  << "{\"status\":\"ok\""
                  << ",\"preprocess_ms\":" << preprocess_ms
                  << ",\"runtime_ms\":" << runtime_ms
                  << ",\"pre_plus_solve_ms\":" << pre_plus_solve_ms
                  << ",\"pre_amortized_ms\":" << pre_amortized_ms
                  << ",\"operations\":" << operations
                  << ",\"valid\":" << (valid ? "true" : "false")
                  << ",\"metadata\":{\"dtype\":\"" << data_type_name() << "\""
                  << ",\"index_type\":\"int32\""
                  << ",\"actual_nnz\":" << matrix.values.size()
                  << ",\"declared_nnz\":" << matrix.declared_nnz
                  << ",\"rhs_columns\":" << rhs_columns
                  << ",\"dense_layout\":\"row-major\""
                  << ",\"op_a\":\"N\",\"op_b\":\"N\""
                  << ",\"alpha\":1,\"beta\":0"
                  << ",\"input_seed\":" << input_seed
                  << ",\"complex_projected_to_real\":"
                  << (matrix.complex_projected_to_real ? "true" : "false")
                  << ",\"max_absolute_error\":";
        write_json_number(std::cout, max_absolute_error);
        std::cout << ",\"max_relative_error\":";
        write_json_number(std::cout, max_relative_error);
        std::cout << ",\"max_normalized_error\":";
        write_json_number(std::cout, max_normalized_error);
        std::cout << ",\"failed_elements\":" << failed_elements
                  << ",\"invalid_elements\":" << invalid_elements
                  << ",\"nan_elements\":" << nan_elements
                  << ",\"inf_elements\":" << inf_elements
                  << ",\"validation_status\":\"" << validation_status << "\""
                  << ",\"validation_rule\":\"dynamic-row-bound\""
                  << ",\"validation_safety_factor\":" << validation_safety_factor
                  << ",\"unit_roundoff\":" << static_cast<double>(unit_roundoff)
                  << ",\"max_row_nnz\":" << max_row_nnz
                  << ",\"reference_precision\":\"long double\""
                  << ",\"output_precision\":\"" << data_type_name() << "\""
                  << ",\"warmup\":" << warmup
                  << ",\"iterations\":" << iterations
                  << ",\"algorithm\":\"" << json_escape(algorithm) << "\""
                  << ",\"library_version\":\"" << json_escape(library_version) << "\""
                  << ",\"gpu_runtime\":\"" << QIWU_GPU_BACKEND << "\""
#if !defined(QIWU_BACKEND_HIP)
                  << ",\"cuda_runtime_version\":" << runtime_version
                  << ",\"cuda_driver_version\":" << driver_version
                  << ",\"cusparse_version\":\"" << cusparse_major << "."
                  << cusparse_minor << "." << cusparse_patch << "\""
#else
                  << ",\"hip_runtime_version\":" << runtime_version
                  << ",\"hip_driver_version\":" << driver_version
#endif
                  << "}}" << std::endl;
            } catch (const std::exception& error) {
                emit_failure(stage, error.what(), destroy_error, rhs_columns);
            } catch (...) {
                emit_failure(stage, "unknown exception", destroy_error, rhs_columns);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        emit_failure(stage, error.what(), destroy_error);
        return 1;
    } catch (...) {
        emit_failure(stage, "unknown exception", destroy_error);
        return 1;
    }
}

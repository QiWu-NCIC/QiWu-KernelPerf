#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceCount failed\n");
        return 1;
    }
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
            continue;
        }
        std::printf(
            "device=%d name=%s arch=%s multiprocessors=%d clock_khz=%d memory_clock_khz=%d bus_bits=%d total_memory=%zu\n",
            device,
            properties.name,
            properties.gcnArchName,
            properties.multiProcessorCount,
            properties.clockRate,
            properties.memoryClockRate,
            properties.memoryBusWidth,
            properties.totalGlobalMem);
    }
    return count > 0 ? 0 : 2;
}

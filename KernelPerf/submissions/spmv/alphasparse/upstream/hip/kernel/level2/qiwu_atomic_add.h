#pragma once

__device__ inline void qiwu_atomic_add_preserve_subnormal(float* address, float value) {
    auto* bits = reinterpret_cast<int*>(address);
    int old = atomicCAS(bits, 0, 0);
    int assumed;
    do {
        assumed = old;
        const float sum = __int_as_float(assumed) + value;
        old = atomicCAS(bits, assumed, __float_as_int(sum));
    } while (old != assumed);
}

template <typename T>
__device__ inline void qiwu_atomic_add_preserve_subnormal(T* address, T value) {
    atomicAdd(address, value);
}

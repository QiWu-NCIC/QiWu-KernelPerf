#pragma once

// The benchmark contract uses CUDA spelling for historical compatibility.  A
// HIP worker defines QIWU_BACKEND_HIP and receives the equivalent HIP runtime
// symbols through this small adapter.  Keeping the mapping in one public
// header lets submissions call the native runtime without duplicating worker
// specific code.
#if defined(QIWU_BACKEND_HIP)
#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStream_t hipStream_t
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion

// CUDA's mask-bearing shuffle spelling is not exposed by DTK's clang front
// end.  Keep CUDA's logical 32-lane reduction semantics on wave64 hardware.
#define __shfl_down_sync(mask, value, delta) __shfl_down(value, delta, 32)

#define QIWU_GPU_BACKEND "HIP"
#else
#include <cuda_runtime.h>
#define QIWU_GPU_BACKEND "CUDA"
#endif

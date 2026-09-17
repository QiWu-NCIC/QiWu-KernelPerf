#pragma once

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
#define QIWU_GPU_BACKEND "HIP"
#else
#include <cuda_runtime.h>
#define QIWU_GPU_BACKEND "CUDA"
#endif

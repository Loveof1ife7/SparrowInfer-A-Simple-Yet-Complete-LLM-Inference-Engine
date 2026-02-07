#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(err) \
  do { \
    cudaError_t err__ = (err); \
    if (err__ != cudaSuccess) { \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
      abort(); \
    } \
  } while (0)

#define CUDA_KERNEL_CHECK() \
  do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)

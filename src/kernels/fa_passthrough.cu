#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void fa_passthrough_kernel(
    const half* __restrict__ q,
    half* __restrict__ out,
    int n)
{    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = q[idx];
    }
}

void launch_fa_passthrough(
    const half* q,
    half* out,
    int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fa_passthrough_kernel<<<blocks, threads>>>(q, out, n);
}